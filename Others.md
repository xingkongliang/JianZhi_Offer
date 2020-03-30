# Others

   * 面试题：[最少硬币数量](#最少硬币数量)
   * 面试题：[TEMP](#TEMP)
   * 面试题：[TEMP](#TEMP)

## 最少硬币数量


带备忘录的递归：自顶向下
```python
from typing import List
class Solution:
    def coinChange(self, coins: List[int], amount: int):
        memo = dict()
        def dp(n):
            if n in memo: return memo[n]

            if n == 0: return 0
            if n < 0: return -1
            res = float('INF')
            for coin in coins:
                subproblem = dp(n-coin)
                if subproblem == -1: continue
                res = min(res, 1+subproblem)
            memo[n] = res if res != float('INF') else -1
            return memo[n]
        return dp(amount)

print(Solution().coinChange([1, 2, 5], 11))
```

dp数组迭代解法：自底向上

```python
from typing import List
class Solution:
    def coinChange(self, coins: List[int], amount: int):
        dp = [amount+1 for _ in range(amount+1)]
        dp[0] = 0
        for i in range(len(dp)):
            for coin in coins:
                if i - coin < 0: continue
                dp[i] = min(dp[i], 1 + dp[i-coin])
        if dp[amount] == amount + 1:
            return -1
        else:
            return dp[amount]

print(Solution().coinChange([1, 2, 5], 11))
```
