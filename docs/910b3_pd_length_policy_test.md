# 910B3 上测试 prefill 分桶和 decode 代价选路

远端实验室主机、SSH、一键脚本与踩坑清单见 [agent_remote_patent_live.md](agent_remote_patent_live.md)（供其他 agent 复用）。

这份说明用来在昇腾 910B3 上比较两套 router 配置：

- 对照：P 和 D 都用 `round_robin`
- 实验：P 用 `prefill_bucket`，D 用 `decode_kv_cost`

比较的是同一份短/长混合请求。短请求的 TTFT 应该下降，长请求不应被送进短桶，两台 decode 的 KV 占用不应一只特别满。

Router 在请求到达时就选定 P 和 D。这次测试不验证「等 prefill 结束再选 decode」。

## 1. 集群怎么配

最小可看出差别的规模是 6 张 910B3，例如一台 8 卡机器留两张备用：

| 角色 | 卡 | HTTP | 长度上限 | 桶 |
|---|---|---|---|---|
| prefill 短 | 0 | `:18001` | 4096 | 0，天花板 2048 |
| prefill 短 | 1 | `:18002` | 4096 | 0 |
| prefill 长 | 2 | `:18003` | 16384 或更大 | 开放桶 |
| prefill 长 | 3 | `:18004` | 16384 或更大 | 开放桶 |
| decode | 4 | `:18011` | 与最长 prompt 相同 | 不分桶 |
| decode | 5 | `:18012` | 与最长 prompt 相同 | 不分桶 |

六台进程使用同一份权重、同一个 `served-model-name`、同一份 tokenizer。P 与 D 之间要能走 Mooncake 的 KV 网络，router 所在机器要能访问上面六个 HTTP 端口，以及两台 D 的 `/metrics`。

910B3 节点是 aarch64。Router 二进制要在同架构的 CPU 上 `cargo build --release`，不要把 Windows 或 x86 编出来的二进制拷过去。

每台进程单独设置 `ASCEND_RT_VISIBLE_DEVICES`。下面是单卡 TP=1 的形状，并行度和模型按实际权重改。Prefill 是 `kv_producer`，decode 是 `kv_consumer`。不要加 `--disable-log-stats`，否则 D 的 `/metrics` 里没有 KV 占用，代价函数会退回只看 router 本地在途数。

```bash
export ASCEND_RT_VISIBLE_DEVICES=0
export HCCL_IF_IP=<本机业务 IP>
export GLOO_SOCKET_IFNAME=<网卡名>
export TP_SOCKET_IFNAME=<网卡名>
export VLLM_USE_V1=1

vllm serve /path/to/model \
  --host 0.0.0.0 --port 18001 \
  --served-model-name test-model \
  --tensor-parallel-size 1 \
  --max-model-len 4096 \
  --max-num-batched-tokens 4096 \
  --gpu-memory-utilization 0.9 \
  --trust-remote-code \
  --kv-transfer-config '{"kv_connector":"MooncakeConnectorV1","kv_role":"kv_producer","kv_port":"30000","kv_connector_extra_config":{"prefill":{"dp_size":1,"tp_size":1},"decode":{"dp_size":1,"tp_size":1}}}'
```

长 prefill 把卡号、端口、`max-model-len` 换成 16384 以上，`kv_port` 每台错开。Decode 把 `kv_role` 改成 `kv_consumer`，`max-model-len` 用长 prefill 的那个值。`kv_port` 是 Mooncake 传输端口，不要和 HTTP 端口撞车。

启动后先确认两件事：

```bash
curl -s http://<prefill-host>:18001/health
curl -s http://<decode-host>:18011/metrics | grep -E 'vllm:kv_cache_usage_perc|vllm:num_requests_running|vllm:num_requests_waiting'
```

第二条必须能看到这三项。没有的话，decode 代价选路不会按 KV 占用工作。

`--kv-connector mooncake` 的 router 启动时会对每台 prefill 发 `GET http://<host>:<bootstrap>/query`，读取每个 DP rank 的 `engine_id`。命令行里 bootstrap 端口写在 prefill URL 后面；不写时用 8998。这个端口不一定等于上面的 `kv_port`。先在 router 机器上执行：

```bash
curl -s http://<prefill-host>:8998/query
```

返回里要有 engine id。如果 8998 没有这个接口，向现网 Mooncake bootstrap 的实际端口重试，并把该端口写进 router 的 `--prefill`。查询失败时 router 会直接退出，不会带着残缺的 KV 参数继续服务。

DP 大于 1 时，router 眼中的实例是 `http://host:port@0` 这种 rank，不是整机一个 URL。本次测试把每张卡当成一个独立进程、DP=1，避免桶成员和 rank 对不齐。

## 2. Router 怎么配

先起对照，再起实验。两次使用同一组 worker、同一份请求。Router 监听 `:8090`。

对照：

```bash
./target/release/vllm-router \
  --host 0.0.0.0 --port 8090 \
  --vllm-pd-disaggregation \
  --kv-connector mooncake \
  --prefill http://10.0.0.1:18001 8998 \
  --prefill http://10.0.0.1:18002 8998 \
  --prefill http://10.0.0.1:18003 8998 \
  --prefill http://10.0.0.1:18004 8998 \
  --decode http://10.0.0.1:18011 \
  --decode http://10.0.0.1:18012 \
  --prefill-policy round_robin \
  --decode-policy round_robin \
  --prometheus-host 0.0.0.0 --prometheus-port 29000
```

实验把最后两个策略换成：

```bash
  --prefill-policy prefill_bucket \
  --decode-policy decode_kv_cost \
  --prefill-bucket-max-tokens 2048 \
  --prefill-bucket-worker http://10.0.0.1:18001=0 \
  --prefill-bucket-worker http://10.0.0.1:18002=0
```

`--prefill-bucket-max-tokens 2048` 产生两个桶：小于等于 2048 的进 0 号桶，更长的进开放桶。没有写进 `--prefill-bucket-worker` 的 `18003` 和 `18004` 就在开放桶。短桶 URL 必须和 `--prefill` 里的 URL 完全一致，包括没有多余的斜杠。

分桶规则：

- 请求长度来自 header `x-prompt-tokens`。没有这个 header 时，router 用 prompt 字节数乘 0.25 估算，中文会偏，910B3 测量时不要依赖估算。
- 带 `x-session-id` 时，在选中的桶里面做 consistent hash。
- 短桶最轻在途数达到 8，并且比长桶最轻在途数高出 1.5 倍，短请求才会溢到长桶。
- 长请求不会进短桶。

Decode 每秒拉取 `http://<decode>/metrics`。占用率达到 0.92，或 waiting 达到 8，这台 D 会被跳过。样本超过 3 秒未更新时，不再使用这份占用率。

启动日志里要有 Mooncake bootstrap 查询成功，以及实验配置下的 `Polling decode KV metrics`。然后：

```bash
curl -s http://127.0.0.1:8090/health
```

## 3. 客户端请求

请求打到 router 的 `/v1/chat/completions`，不要打到某一台 P 或 D。

每条请求需要：

- `model` 等于 `--served-model-name`
- `stream: true`，方便记录首 token 时间
- `max_tokens: 64`，让 decode 的差异主要来自 prompt KV，而不是超长输出
- `temperature: 0`
- header `x-prompt-tokens` 等于这条 prompt 的真实 token 数
- 测会话粘滞时再加 `x-session-id`

顺序是短、长交替。先各发 4 轮预热，再记 30 轮。短请求用 256 token，长请求用 8192 token。8192 会越过 2048 的桶边界，又落在长 prefill 的 `max-model-len` 里面。

客户端脚本是 `router/scripts/910b3/length_mix_client.py`。在能访问 router 的机器上，用服务器同一份 tokenizer：

```bash
python3 router/scripts/910b3/length_mix_client.py \
  --url http://<router>:8090/v1/chat/completions \
  --model test-model \
  --tokenizer /path/to/model \
  --short-tokens 256 \
  --long-tokens 8192 \
  --rounds 30 \
  --warmup 4 \
  --max-tokens 64
```

对照和实验各跑一次，保留两次打印的 `ttft_p50_s` 和 `ttft_p90_s`。

手工抽查一条长请求有没有进对的桶：

```bash
curl -N http://<router>:8090/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -H 'x-prompt-tokens: 8192' \
  -H 'x-session-id: long-session' \
  -d '{"model":"test-model","messages":[{"role":"user","content":"<与 8192 token 一致的正文>"}],"max_tokens":64,"temperature":0,"stream":true}'
```

Router 日志里这条的 prefill 应是 `18003` 或 `18004`。把 `x-prompt-tokens` 改成 256 后，prefill 应是 `18001` 或 `18002`。同一个 `x-session-id` 连续两次，桶内的那台 P 应相同。

## 4. 其他事项

- 两组实验之间等 D 的 `/metrics` 里 running 回到 0，再开始下一组。不要在上一组的长 KV 还占着时切策略。
- 只改 router 策略。模型、并行、`max-model-len`、Mooncake 端口和请求文件保持不变。
- 短桶机器的 `max-model-len` 必须大于等于 2048。开放桶和两台 D 必须大于等于本次最长 prompt 加 `max_tokens`。
- 客户端打印的 token 数以 tokenizer 编码结果为准。脚本在提供 `--tokenizer` 时会用截断后的真实长度覆盖 header。
- 有效的样子：实验组短请求 TTFT 的 p50 明显低于对照；router 日志里 8192 的请求没有出现在 18001/18002；两台 D 的 `vllm:kv_cache_usage_perc` 差距小于对照。长请求 TTFT 可以持平或略差，因为长桶只有两张卡。
- 短请求 TTFT 没有差别时，先看是不是所有请求都溢到了开放桶，以及 D 的 metrics 是不是空的。溢出会在短桶在途数过高时发生。
- 单张卡、或者全部 P 都在开放桶时，这个策略不会带来排队收益。
- 结果是这次混合流量下的端到端 TTFT，不是离线模拟里的 tick。
