## MinLoRA
A minimal reproduction of GPT2 + LoRA.

### Observations
Model FLOPS utilization increases when using float16.

> Using float32 (around the same time that nan losses emerge for float16, loss and gradients explode)
```
iter 0: loss 10.9264, time 95130.82ms, mfu -100.00%
iter 10: loss 9.8973, time 1516.37ms, mfu 14.60%
iter 20: loss 9.0391, time 1515.06ms, mfu 14.60%
iter 30: loss 7.8356, time 1521.29ms, mfu 14.59%
iter 40: loss 6.6008, time 1529.10ms, mfu 14.58%
iter 50: loss 6.0871, time 1525.52ms, mfu 14.57%
iter 60: loss 5.8614, time 1529.32ms, mfu 14.56%
iter 70: loss 5.6643, time 1537.46ms, mfu 14.55%
iter 80: loss 5.3005, time 1531.56ms, mfu 14.54%
iter 90: loss 5.0595, time 1529.68ms, mfu 14.53%
iter 100: loss 4.8186, time 1537.05ms, mfu 14.52%
iter 110: loss 270.3931, time 1464.67ms, mfu 14.58%
iter 120: loss 182.6521, time 1462.69ms, mfu 14.63%
iter 130: loss 121.9155, time 1461.20ms, mfu 14.68%
iter 140: loss 96.6580, time 1465.78ms, mfu 14.73%
iter 150: loss 68.6309, time 1481.96ms, mfu 14.75%
iter 160: loss 78.4303, time 1480.53ms, mfu 14.77%
iter 170: loss 40.0218, time 1527.70ms, mfu 14.74%
iter 180: loss 31.0113, time 1524.71ms, mfu 14.72%
```

> Using float16 (but for some reason, nan losses emerge)
```
iter 0: loss 10.9264, time 33110.45ms, mfu -100.00%
iter 10: loss 9.8973, time 460.33ms, mfu 48.08%
iter 20: loss 9.0391, time 460.54ms, mfu 48.08%
iter 30: loss 7.8357, time 458.71ms, mfu 48.10%
iter 40: loss 6.6009, time 458.29ms, mfu 48.12%
iter 50: loss 6.0871, time 459.38ms, mfu 48.12%
iter 60: loss 5.8613, time 461.50ms, mfu 48.11%
iter 70: loss 5.6641, time 460.52ms, mfu 48.10%
iter 80: loss 5.3016, time 460.07ms, mfu 48.10%
iter 90: loss 5.0541, time 461.03ms, mfu 48.09%
iter 100: loss 4.8292, time 460.11ms, mfu 48.10%
iter 110: loss nan, time 440.71ms, mfu 48.31%
iter 120: loss nan, time 441.36ms, mfu 48.49%
iter 130: loss nan, time 441.87ms, mfu 48.65%
iter 140: loss nan, time 441.00ms, mfu 48.81%
iter 150: loss nan, time 442.14ms, mfu 48.93%
iter 160: loss nan, time 444.17ms, mfu 49.02%
iter 170: loss nan, time 441.49ms, mfu 49.13%
iter 180: loss nan, time 441.53ms, mfu 49.23%
```
