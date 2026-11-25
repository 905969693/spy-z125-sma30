# 冻结规则：四资产相关速度 Z>1.25 + SPY 30 日均线门控

这是 VY 于 2026-11-20（Asia/Hong_Kong）冻结的 SPY 满仓 / 空仓规则的实现。完整工作定义见 [docs/冻结规则_Z125_SPY30日均线.md](docs/冻结规则_Z125_SPY30日均线.md)。

这套规则不是预注册检验，也不是已经验证的最优参数。阈值 1.25 和 30 日均线是在看过其他 Z 阈值和 5 / 10 / 30 日门控之后才固定的，存在选择偏差。下面的正式全样本数字是 0 成本结果，不能当成样本外验证。

除非明确要求，不要改篮子、复合公式、窗口、阈值或门控长度。

## 规则（已冻结）

- 篮子：SPY、TLT、GLD、DXY（ICE 美元指数，yfinance 代码 `DX-Y.NYB`）。
- 对日收益做 63 个交易日滚动 Pearson 相关，`min_periods=63`，

  `s = -corr(SPY,TLT) - corr(SPY,DXY) + corr(SPY,GLD) + corr(TLT,DXY)`。
- `Δs_t = s_t - s_{t-21}`。`μ`、`σ` 只用 t 之前 180 个交易日的 `Δs`（`dx.shift(1).rolling(180, min_periods=180)`），标准差 `ddof=1`，`Z_t = (Δs_t - μ_t) / σ_t`。
- 仅当 SPY 收盘价**严格低于**其 30 日简单均值（含当日；相等不算低于）时门控打开。
- 若 `Z_t > 1.25` 且门控打开，当日收盘信号为空仓，否则满仓 SPY。信号在 t 日收盘确认，**t+1 日**生效。空仓持有 BIL 总回报。交易成本按 0。

SPY 与 BIL 的日收益按各自交易日历计算，再对齐到四资产交集。因此像 2016-10-10、2016-11-11 这种 SPY 有行情、交集里没有的交易日，仍会进入次一个对齐日的 SPY 收益。这与冻结时使用的官方日序列一致。


## 正式全样本（0 成本，供对照，不是验证）

样本：2007-06-01 至 2026-11-13，n = 4894。净值在 2007-05-31 设为 1。

| 项目 | 冻结规则 | 买入持有 SPY |
| --- | --- | --- |
| 期末净值 | 9.49 | 7.00 |
| 年化收益 | 12.28% | 10.54% |
| 年化波动 | 17.63% | 19.68% |
| Sharpe（rf=0） | 0.75 | 0.61 |
| 最大回撤 | −41.4% | −55.2% |
| 空仓时间占比 | 4.1%（203 天） | 0 |
| 仓位切换次数 | 113 | 0 |

年化交易日 252；波动与 Sharpe 用样本标准差 `ddof=1`；Sharpe 的无风险利率为 0。区间收益是 `P_end / P_start - 1`，不含起点当日收益。

## 安装与运行

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

复现上面这张正式全样本表，请用仓库里冻结的价格快照（不要用后来可能被复权修正的缓存）：

```bash
spy-z125-sma30 --official-sample backtest --windows
spy-z125-sma30 --official-sample signal
```

如果本机已有只读价格缓存（列名含 `Date` 与 `Close` 的 CSV），把它们放到一个目录，文件名为 `SPY_max.csv`、`TLT_max.csv`、`GLD_max.csv`、`DXY_max.csv`、`BIL_max.csv`：

```bash
spy-z125-sma30 --cache-dir /path/to/cache backtest --windows
spy-z125-sma30 --cache-dir /path/to/cache signal
```

没有缓存时，用 yfinance 下载（`auto_adjust=True`，Close 视为复权价）。新数据只会写到 `--write-cache-dir`（默认 `./cache`），不会覆盖已有的官方研究缓存：

```bash
spy-z125-sma30 --download backtest --windows
spy-z125-sma30 --download signal
```

`backtest` 默认截到正式样本终点 2026-11-13，便于对照上表。要算到数据的最后一天，加上 `--end YYYY-MM-DD`。`signal` 默认用数据里最近一个已形成的收盘信号，次日生效。

本仓库只计算信号和回测，不会下单或改仓。

## 测试

```bash
pytest
```

在能读到官方价格缓存的机器上，测试会把正式全样本结果对到上表背后的精确数字，并与已发布的 `gate_z_grid_daily.csv` 对齐。没有这些文件时，只跑不依赖缓存的因果性和次日生效测试。

## 另一套信号：EWMA（未冻结，不是本规则的替代）

[`ewma-sma30/`](ewma-sma30/README.md) 是单独的一套 SPY 空仓 / 满仓信号（对数收益的 EWMA 符号协方差，SMA30，λ=0.94，5 日百分位回落 −0.70）。它**没有冻结**，**不是**预注册或样本外验证过的最优参数，也**不是**上面这套 Z>1.25 + SMA30 Pearson 规则的替代或修改。两套实现互不引用。说明和运行方式见该目录的 README。
