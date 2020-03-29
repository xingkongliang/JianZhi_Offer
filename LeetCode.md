# LeetCode

   * 面试题：[5-最长回文子串](#5-最长回文子串)
   * 面试题：[32-最长有效括号](#32-最长有效括号)
   * 面试题：[62-不同路径](#62-不同路径)
   * 面试题：[63-不同路径II](#63-不同路径II)
   * 面试题：[64-最小路径和](#64-最小路径和)
   * 面试题：[72-编辑距离](#72-编辑距离)
   * 面试题：[84-柱状图中最大的矩形](#84-柱状图中最大的矩形)
   * 面试题：[85-最大矩形](#85-最大矩形)
   * 面试题：[TEMP](#TEMP)
   * 面试题：[TEMP](#TEMP)

   * 面试题：[通配符匹配](#通配符匹配)
   * 面试题：[跳跃游戏II](#跳跃游戏II)
   * 面试题：[加油站](#加油站)
   * 面试题：[135-分发糖果](#135-分发糖果)

   * 面试题：[202-快乐数](#202-快乐数)
   * 面试题：[203-移除链表元素](#203-移除链表元素)
   * 面试题：[204-计数质数](#204-计数质数)
   * 面试题：[205-同构字符串](#205-同构字符串)
   * 面试题：[206-反转链表](#206-反转链表)
   * 面试题：[219-存在重复元素II](#219-存在重复元素II)
   * 面试题：[231-2的幂](#231-2的幂)
   * 面试题：[234-回文链表](#234-回文链表)
   * 面试题：[235-二叉搜索树的最近公共祖先](#235-二叉搜索树的最近公共祖先)
   * 面试题：[242-有效的字母异位词](#242-有效的字母异位词)
   * 面试题：[257-二叉树的所有路径](#257-二叉树的所有路径)
   * 面试题：[258-各位相加](#258-各位相加)
   * 面试题：[268-缺失数字](#268-缺失数字)
   * 面试题：[283-移动零](#283-移动零)
   * 面试题：[TEMP](#TEMP)
   * 面试题：[TEMP](#TEMP)
   * 面试题：[TEMP](#TEMP)
   * 面试题：[TEMP](#TEMP)
   * 面试题：[TEMP](#TEMP)
   * 面试题：[TEMP](#TEMP)


## 5-最长回文子串
给定一个字符串 s，找到 s 中最长的回文子串。你可以假设 s 的最大长度为 1000。

示例 1：

输入: "babad"
输出: "bab"
注意: "aba" 也是一个有效答案。
示例 2：

输入: "cbbd"
输出: "bb"

### 解法
扩展中心

我们知道回文串一定是对称的，所以我们可以每次循环选择一个中心，进行左右扩展，判断左右字符是否相等即可。

![5-最长回文子串-4](https://pic.leetcode-cn.com/1b9bfe346a4a9a5718b08149be11236a6db61b3922265d34f22632d4687aa0a8-image.png)

由于存在奇数的字符串和偶数的字符串，所以我们需要从一个字符开始扩展，或者从两个字符之间开始扩展，所以总共有 n+n-1 个中心。

```python
class Solution:
    def expandAroundCenter(self, s: str, left: int, right:int) -> str:
        L = left; R = right
        while L >= 0 and R < len(s) and s[L] == s[R]:
            L -= 1
            R += 1
        return R - L - 1

    def longestPalindrome(self, s: str) -> str:
        if s == None or len(s) < 1:
            return ""
        start = 0; end = 0
        for i in range(len(s)):
            len1 = self.expandAroundCenter(s, i, i)
            len2 = self.expandAroundCenter(s, i, i + 1)
            len_ = max(len1, len2)
            if len_ > end - start:
                start = i - (len_ - 1) // 2
                end = i + len_ // 2
        return s[start:end + 1]
```

## 32-最长有效括号
给定一个只包含 '(' 和 ')' 的字符串，找出最长的包含有效括号的子串的长度。

示例 1:
```
输入: "(()"
输出: 2
解释: 最长有效括号子串为 "()"
```

示例 2:
```
输入: ")()())"
输出: 4
解释: 最长有效括号子串为 "()()"
```

```python
class Solution:
    def longestValidParentheses(self, s: str) -> int:
        maxans = 0
        length = len(s)
        dp = [0 for _ in range(length)]  # 存放最长有效字符串的长度
        for i in range(1, length):
            if s[i] == ')':
                # ...()
                if s[i-1] == '(':
                    if i >= 2:
                        dp[i] = dp[i-2] + 2  # 前一个有效字符串长度 + 2
                    else:
                        dp[i] = 2
                # ...))
                elif i - dp[i-1] > 0 and s[i-dp[i-1]-1] == '(':
                    if i - dp[i-1] >= 2:
                        dp[i] = dp[i-1] + dp[i-dp[i-1]-2] + 2
                    else:
                        # (()())
                        dp[i] = dp[i-1] + 2
            maxans = max(maxans, dp[i])
        return maxans
```

## 62-不同路径

一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

问总共有多少条不同的路径？

### 解法1：排列组合
因为机器到底右下角，向下几步，向右几步都是固定的，

比如，m=3, n=2，我们只要向下 1 步，向右 2 步就一定能到达终点。

所以有 $C^{m-1}_{m+n-2}$
​

```python
import math
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        ans = math.factorial(n+m-2)/(math.factorial(n-1)*math.factorial(m-1))
        return int(ans)
```
### 解法2：动态规划
我们令 `dp[i][j]` 是到达 `i`, `j` 最多路径

动态方程：`dp[i][j] = dp[i-1][j] + dp[i][j-1]`

注意，对于第一行 `dp[0][j]`，或者第一列 `dp[i][0]`，由于都是在边界，所以只能为 1

```python
class Solution:
    def uniquePaths(self, m: int, n: int) -> int:
        dp = [[1]*n] + [[1]+[0] * (n-1) for _ in range(m-1)]
        #print(dp)
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = dp[i-1][j] + dp[i][j-1]
        return dp[-1][-1]
```

## 63-不同路径II
一个机器人位于一个 m x n 网格的左上角 （起始点在下图中标记为“Start” ）。

机器人每次只能向下或者向右移动一步。机器人试图达到网格的右下角（在下图中标记为“Finish”）。

现在考虑网格中有障碍物。那么从左上角到右下角将会有多少条不同的路径？



网格中的障碍物和空位置分别用 1 和 0 来表示。

说明：m 和 n 的值均不超过 100。

示例 1:
```
输入:
[
  [0,0,0],
  [0,1,0],
  [0,0,0]
]
输出: 2
解释:
3x3 网格的正中间有一个障碍物。
从左上角到右下角一共有 2 条不同的路径：
1. 向右 -> 向右 -> 向下 -> 向下
2. 向下 -> 向下 -> 向右 -> 向右
```

```python
class Solution:
    def uniquePathsWithObstacles(self, obstacleGrid: List[List[int]]) -> int:
        R = len(obstacleGrid)
        C = len(obstacleGrid[0])

        if obstacleGrid[0][0] == 1:
            return 0

        obstacleGrid[0][0] = 1
        for i in range(1, R):
            if obstacleGrid[i][0] == 0 and obstacleGrid[i-1][0] == 1:
                obstacleGrid[i][0] = 1
            else:
                obstacleGrid[i][0] = 0

        for i in range(1, C):
            if obstacleGrid[0][i] == 0 and obstacleGrid[0][i-1] == 1:
                obstacleGrid[0][i] = 1
            else:
                obstacleGrid[0][i] = 0

        for i in range(1, R):
            for j in range(1, C):
                if obstacleGrid[i][j] == 0:
                    obstacleGrid[i][j] = obstacleGrid[i-1][j] + obstacleGrid[i][j-1]
                else:
                    # 如果当前格子被置为1，则把累积的该位置置0
                    obstacleGrid[i][j] = 0

        return obstacleGrid[R-1][C-1]
```

## 64-最小路径和

给定一个包含非负整数的 m x n 网格，请找出一条从左上角到右下角的路径，使得路径上的数字总和为最小。

说明：每次只能向下或者向右移动一步。

示例:
```
输入:
[
  [1,3,1],
  [1,5,1],
  [4,2,1]
]
输出: 7
解释: 因为路径 1→3→1→1→1 的总和最小。
```

```python
class Solution:
    def minPathSum(self, grid: List[List[int]]) -> int:
        R = len(grid)
        C = len(grid[0])

        for i in range(R-2, -1, -1):
            grid[i][C-1] += grid[i+1][C-1]

        for j in range(C-2, -1, -1):
            grid[R-1][j] += grid[R-1][j+1]

        for i in range(R-2, -1, -1):
            for j in range(C-2, -1, -1):
                if grid[i+1][j] > grid[i][j+1]:
                    grid[i][j] += grid[i][j+1]
                else:
                    grid[i][j] += grid[i+1][j]
        return grid[0][0]
```

## 72-编辑距离

给定两个单词 word1 和 word2，计算出将 word1 转换成 word2 所使用的最少操作数 。

你可以对一个单词进行如下三种操作：

插入一个字符
删除一个字符
替换一个字符
示例 1:
```
输入: word1 = "horse", word2 = "ros"
输出: 3
解释:
horse -> rorse (将 'h' 替换为 'r')
rorse -> rose (删除 'r')
rose -> ros (删除 'e')
```

示例 2:
```
输入: word1 = "intention", word2 = "execution"
输出: 5
解释:
intention -> inention (删除 't')
inention -> enention (将 'i' 替换为 'e')
enention -> exention (将 'n' 替换为 'x')
exention -> exection (将 'n' 替换为 'c')
exection -> execution (插入 'u')
```

### 解析：

如果两个子串的最后一个字母相同，word1[i] = word2[i] 的情况下：
$$
D[i][j]=1+min(D[i−1][j],D[i][j−1],D[i−1][j−1]−1)
D[i][j]=1+min(D[i−1][j],D[i][j−1],D[i−1][j−1]−1)
$$
否则，word1[i] != word2[i] 我们将考虑替换最后一个字符使得他们相同：
$$
D[i][j]=1+min(D[i−1][j],D[i][j−1],D[i−1][j−1])
D[i][j]=1+min(D[i−1][j],D[i][j−1],D[i−1][j−1])
$$

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        m = len(word1)
        n = len(word2)
        if m == 0 or n == 0:
            return m + n

        d = [[0]*(m+1) for _ in range(n+1)]
        for i in range(n+1):
            d[i][0] = i
        for j in range(m+1):
            d[0][j] = j
        for i in range(1, n+1):
            for j in range(1, m+1):
                if word2[i-1] == word1[j-1]:
                    d[i][j] = 1 + min(d[i-1][j-1]-1, d[i-1][j], d[i][j-1])
                else:
                    d[i][j] = 1 + min(d[i-1][j-1], d[i-1][j], d[i][j-1])
        return d[n][m]
```

## 84-柱状图中最大的矩形

给定 n 个非负整数，用来表示柱状图中各个柱子的高度。每个柱子彼此相邻，且宽度为 1 。

求在该柱状图中，能够勾勒出来的矩形的最大面积。


![84-柱状图中最大的矩形-1](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/histogram.png)


以上是柱状图的示例，其中每个柱子的宽度为 1，给定的高度为 [2,1,5,6,2,3]。


![84-柱状图中最大的矩形-2](https://assets.leetcode-cn.com/aliyun-lc-upload/uploads/2018/10/12/histogram_area.png)


图中阴影部分为所能勾勒出的最大矩形面积，其面积为 10 个单位。



示例:
```
输入: [2,1,5,6,2,3]
输出: 10
```


```python
class Solution:
    def largestRectangleArea(self, heights: List[int]) -> int:
        stack = [-1]
        maxarea = 0
        for i in range(len(heights)):
            while stack[-1] != -1 and heights[stack[-1]] >= heights[i]:
                last_one = stack.pop()
                maxarea = max(maxarea, heights[last_one] * (i - stack[-1] - 1))
            stack.append(i)
        while stack[-1] != -1:
            last_one = stack.pop()
            maxarea = max(maxarea, heights[last_one] * (len(heights) - stack[-1] - 1))
        return maxarea
```

## 85-最大矩形

给定一个仅包含 0 和 1 的二维二进制矩阵，找出只包含 1 的最大矩形，并返回其面积。

示例:
```
输入:
[
  ["1","0","1","0","0"],
  ["1","0","1","1","1"],
  ["1","1","1","1","1"],
  ["1","0","0","1","0"]
]
输出: 6
```

```python
class Solution:

    # Get the maximum area in a histogram given its heights
    def leetcode84(self, heights):
        stack = [-1]

        maxarea = 0
        for i in range(len(heights)):

            while stack[-1] != -1 and heights[stack[-1]] >= heights[i]:
                maxarea = max(maxarea, heights[stack.pop()] * (i - stack[-1] - 1))
            stack.append(i)

        while stack[-1] != -1:
            maxarea = max(maxarea, heights[stack.pop()] * (len(heights) - stack[-1] - 1))
        return maxarea


    def maximalRectangle(self, matrix: List[List[str]]) -> int:

        if not matrix: return 0

        maxarea = 0
        dp = [0] * len(matrix[0])
        for i in range(len(matrix)):
            for j in range(len(matrix[0])):

                # update the state of this row's histogram using the last row's histogram
                # by keeping track of the number of consecutive ones

                dp[j] = dp[j] + 1 if matrix[i][j] == '1' else 0

            # update maxarea with the maximum area from this row's histogram
            maxarea = max(maxarea, self.leetcode84(dp))
        return maxarea
```

## 通配符匹配

给定一个字符串 (s) 和一个字符模式 (p) ，实现一个支持 '?' 和 '\*' 的通配符匹配。

'?' 可以匹配任何单个字符。
'\*' 可以匹配任意字符串（包括空字符串）。
两个字符串完全匹配才算匹配成功。

说明:

s 可能为空，且只包含从 a-z 的小写字母。
p 可能为空，且只包含从 a-z 的小写字母，以及字符 ? 和 \*。

```python
class Solution:
    def isMatch(self, s, p):
        s_len = len(s)
        p_len = len(p)

        # base cases
        if p == s or p == '*':
            return True
        if p == '' or s == '':
            return False

        # init all matrix except [0][0] element as False
        d = [ [False] * (s_len + 1) for _ in range(p_len + 1)]
        d[0][0] = True

        # DP compute
        for p_idx in range(1, p_len + 1):
            # the current character in the pattern is '*'
            if p[p_idx - 1] == '*':
                s_idx = 1
                # d[p_idx - 1][s_idx - 1] is a string-pattern match
                # on the previous step, i.e. one character before.
                # Find the first idx in string with the previous math.
                while not d[p_idx - 1][s_idx - 1] and s_idx < s_len + 1:
                    s_idx += 1
                # If (string) matches (pattern),
                # when (string) matches (pattern)* as well
                d[p_idx][s_idx - 1] = d[p_idx - 1][s_idx - 1]
                # If (string) matches (pattern),
                # when (string)(whatever_characters) matches (pattern)* as well
                while s_idx < s_len + 1:
                    d[p_idx][s_idx] = True
                    s_idx += 1
            # the current character in the pattern is '?'
            elif p[p_idx - 1] == '?':
                for s_idx in range(1, s_len + 1):
                    d[p_idx][s_idx] = d[p_idx - 1][s_idx - 1]
            # the current character in the pattern is not * or '?'
            else:
                for s_idx in range(1, s_len + 1):
                    # Match is possible if there is a previous match
                    # and current characters are the same
                    d[p_idx][s_idx] = \
                    d[p_idx - 1][s_idx - 1] and p[p_idx - 1] == s[s_idx - 1]  

        return d[p_len][s_len]
```

## 跳跃游戏II

给定一个非负整数数组，你最初位于数组的第一个位置。

数组中的每个元素代表你在该位置可以跳跃的最大长度。

你的目标是使用最少的跳跃次数到达数组的最后一个位置。

示例:

输入: [2,3,1,1,4]
输出: 2
解释: 跳到最后一个位置的最小跳跃数是 2。
     从下标为 0 跳到下标为 1 的位置，跳 1 步，然后跳 3 步到达数组的最后一个位置。
说明:

假设你总是可以到达数组的最后一个位置。

```python
class Solution:
    def jump(self, nums: List[int]) -> int:
        end = 0
        maxPosition = 0
        steps = 0
        for i in range(0, len(nums)-1):  # 找能跳的最远的
            maxPosition = max(maxPosition, nums[i]+i)
            if i == end:  # 遇到边界，就更新边界，并且步数加一
                end = maxPosition
                steps += 1
        return steps
```

## 加油站

在一条环路上有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。

你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。

如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。

说明:

如果题目有解，该答案即为唯一答案。
输入数组均为非空数组，且长度相同。
输入数组中的元素均为非负数。

```python
class Solution:
    def canCompleteCircuit(self, gas: List[int], cost: List[int]) -> int:
        n = len(gas)

        total_tank, curr_tank = 0, 0
        starting_station = 0
        for i in range(n):
            total_tank += gas[i] - cost[i]
            curr_tank += gas[i] - cost[i]
            # If one couldn't get here,
            if curr_tank < 0:
                # Pick up the next station as the starting one.
                starting_station = i + 1
                # Start with an empty tank.
                curr_tank = 0

        return starting_station if total_tank >= 0 else -1
```

## 135-分发糖果

老师想给孩子们分发糖果，有 N 个孩子站成了一条直线，老师会根据每个孩子的表现，预先给他们评分。

你需要按照以下要求，帮助老师给这些孩子分发糖果：

每个孩子至少分配到 1 个糖果。
相邻的孩子中，评分高的孩子必须获得更多的糖果。
那么这样下来，老师至少需要准备多少颗糖果呢？

```python
class Solution:
    def candy(self, ratings: List[int]) -> int:
        left = [1 for i in range(len(ratings))]
        right = left[:]

        for i in range(1, len(ratings)):
            if ratings[i] > ratings[i-1]:
                left[i] = left[i-1] + 1
        count = left[-1]
        for i in range(len(ratings)-2, -1, -1):
            if ratings[i] > ratings[i+1]:
                right[i] = right[i+1] + 1
            count += max(left[i], right[i])
        return count
```

## 去除重复字母

给你一个仅包含小写字母的字符串，请你去除字符串中重复的字母，使得每个字母只出现一次。需保证返回结果的字典序最小（要求不能打乱其他字符的相对位置）。

示例 1:

输入: "bcabc"
输出: "abc"
示例 2:

输入: "cbacdcbc"
输出: "acdb"

https://leetcode-cn.com/problems/remove-duplicate-letters/solution/zhan-by-liweiwei1419/

- 1、遍历字符串里的字符，如果读到的字符的 ASCII 值是升序，依次存到一个栈中；
- 2、如果读到的字符在栈中已经存在，这个字符我们不需要；
- 3、如果读到的 ASCII 值比栈顶元素严格小，看看栈顶元素在后面是否还会出现，如果还会出现，则舍弃栈顶元素，而选择后出现的那个字符，这样得到的字典序更小。

```python
class Solution:
    def removeDuplicateLetters(self, s: str) -> str:
        size = len(s)

        last_appear_index = [0 for _ in range(26)]
        if_in_stack = [False for _ in range(26)]

        # 记录每个字符最后一次出现的索引
        for i in range(size):
            last_appear_index[ord(s[i]) - ord('a')] = i

        stack = []
        for i in range(size):
            if if_in_stack[ord(s[i]) - ord('a')]:
                continue

            while stack and ord(stack[-1]) > ord(s[i]) and \
                    last_appear_index[ord(stack[-1]) - ord('a')] >= i:
                top = stack.pop()
                if_in_stack[ord(top) - ord('a')] = False

            stack.append(s[i])
            if_in_stack[ord(s[i]) - ord('a')] = True

        return ''.join(stack)
```

## 生产口罩

[题目](https://www.nowcoder.com/questionTerminal/f0ebe2aa29da4a4a9d0ea71357cf2f91)

[题解](https://blog.nowcoder.net/n/b64a95632bc94939b83f5a1999eb81d0)

考虑动态规划，dp[i][j]dp[i][j]代表选了1−i的工厂后，用j个人所能生产的最多的口罩个数。那么dpdp转移方程式就很简单了,假设我们现在枚举到了第i个人的第hh种策略，其中$a_{ih}$ 代表人数，$b_{ih}$ 代表生产的口罩数。dp[i][j+$a_{ih}$]=max(dp[i][j+$a_{ih}$], dp[i−1][j]+$y_{ih}$)，需要注意的是第i个生产线也可以选择不选任何策略。

```python
class Point:
    def __init__(self, a=0, b=0):
        self.x = a
        self.y = b

class Solution:
    def producemask(self , n , m , strategy):
        """
        n: 生产线数量
        m: 人数
        strategy: 生产线情况
        """
        dp = [[0]*(m+1) for _ in range(n+1)]  # dp[i][j]代表选了i-1的生产线后，用j个人所能生产的最多的口罩个数
        for i in range(1, n+1):  # 第i个生产线
            for j in range(m+1):  # 第j个员工
                for x in strategy[i-1]:  # 从strategy选取策略为x
                    print("生产线: {}, 员工: {}, 策略 人数-口罩数: {}-{}".format(i, j, x.x, x.y))
                    if x.x + j > m:
                        continue
                    dp[i][j+x.x] = max(dp[i][j+x.x], dp[i-1][j]+x.y)
            for j in range(m+1):
                dp[i][j] = max(dp[i][j], dp[i-1][j])
        return dp[n][m]

print(Solution().producemask(3,
                             5,
                             [[Point(1,3),Point(2,4)],
                             [Point(3,4),Point(4,4)],
                             [Point(8,8)]]))
```

## 202-快乐数

编写一个算法来判断一个数是不是“快乐数”。

一个“快乐数”定义为：对于一个正整数，每一次将该数替换为它每个位置上的数字的平方和，然后重复这个过程直到这个数变为 1，也可能是无限循环但始终变不到 1。如果可以变为 1，那么这个数就是快乐数。

```python
class Solution:
    def isHappy(self, n: int) -> bool:
        issam = [n]
        while n != 1:
            a = list(str(n))
            n = 0
            for i in a:
                n += pow(int(i),2)
            issam.append(n)
            if len(issam) != len(set(issam)):
                return False
        return True

作者：i-want-to-see-you
```

## 203-移除链表元素
删除链表中等于给定值 val 的所有节点。

```python
class Solution:
    def removeElements(self, head: ListNode, val: int) -> ListNode:
        dump = ListNode(-1)
        dump.next = head
        prev, curr = dump, head

        while curr:
            if curr.val == val:
                prev.next = curr.next
            else:
                prev = curr
            curr = curr.next
        return dump.next
```

## 204-计数质数
统计所有小于非负整数 n 的质数的数量。

```python
class Solution:
    def countPrimes(self, n: int) -> int:
        isPrim = [ True for _ in range(n)]
        for i in range(2, n):
            if isPrim[i]:
                for j in range(2*i, n, i):
                    isPrim[j] = False
        count = 0
        for i in range(2, n):
            if isPrim[i]:
                count += 1
        return count
```

## 205-同构字符串
给定两个字符串 s 和 t，判断它们是否是同构的。

如果 s 中的字符可以被替换得到 t ，那么这两个字符串是同构的。

所有出现的字符都必须用另一个字符替换，同时保留字符的顺序。两个字符不能映射到同一个字符上，但字符可以映射自己本身。

```python
class Solution:
    def isIsomorphic(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        return self.isIsomorphicHelper(s, t) and self.isIsomorphicHelper(t, s)
    def isIsomorphicHelper(self, s: str, t: str) -> bool:
        s = list(s); t = list(t)
        map = {}
        for i in range(len(s)):
            if s[i] not in map.keys():
                map[s[i]] = t[i]
            if map[s[i]] != t[i]:
                return False
        return True
```

## 206-反转链表
反转一个单链表。

```python
class Solution:
    def reverseList(self, head: ListNode) -> ListNode:
        prev = None
        curr = head
        while curr != None:
            nextTemp = curr.next
            curr.next = prev
            prev = curr
            curr = nextTemp
        return prev
```

## 219-存在重复元素II
给定一个整数数组和一个整数 k，判断数组中是否存在两个不同的索引 i 和 j，使得 `nums[i] = nums[j]`，并且 i 和 j 的差的 绝对值 至多为 k。

```python
class Solution:
    def containsNearbyDuplicate(self, nums: List[int], k: int) -> bool:
        stack = {}
        for i in range(len(nums)):
            if nums[i] not in stack.keys():
                stack[nums[i]] = i
            else:
                if i - stack[nums[i]] <= k:
                    return True
                else:
                    stack[nums[i]] = i
        return False
```

## 231-2的幂
给定一个整数，编写一个函数来判断它是否是 2 的幂次方。

```python
class Solution:
    def isPowerOfTwo(self, n: int) -> bool:
        if n == 0:
            return False
        return n & (n-1) == 0
```

## 234-回文链表
请判断一个链表是否为回文链表。

```python
class Solution:
    def isPalindrome(self, head: ListNode) -> bool:
        if head is None:
            return True

        # Find the end of first half and reverse second half.
        first_half_end = self.end_of_first_half(head)
        second_half_start = self.reverse_list(first_half_end.next)

        # Check whether or not there's a palindrome.
        result = True
        first_position = head
        second_position = second_half_start
        while result and second_position is not None:
            if first_position.val != second_position.val:
                result = False
            first_position = first_position.next
            second_position = second_position.next

        # Restore the list and return the result.
        first_half_end.next = self.reverse_list(second_half_start)
        return result    

    def end_of_first_half(self, head):
        fast = head
        slow = head
        while fast.next is not None and fast.next.next is not None:
            fast = fast.next.next
            slow = slow.next
        return slow

    def reverse_list(self, head):
        previous = None
        current = head
        while current is not None:
            next_node = current.next
            current.next = previous
            previous = current
            current = next_node
        return previous

作者：LeetCode
链接：https://leetcode-cn.com/problems/palindrome-linked-list/solution/hui-wen-lian-biao-by-leetcode/
来源：力扣（LeetCode）
```

## 235-二叉搜索树的最近公共祖先
给定一个二叉搜索树, 找到该树中两个指定节点的最近公共祖先。

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        if p.val > root.val and q.val > root.val:
            return self.lowestCommonAncestor(root.right, p, q)
        elif p.val < root.val and q.val < root.val:
            return self.lowestCommonAncestor(root.left, p, q)
        else:
            return root
```

## 242-有效的字母异位词

给定两个字符串 s 和 t ，编写一个函数来判断 t 是否是 s 的字母异位词。

```python
class Solution:
    def isAnagram(self, s: str, t: str) -> bool:
        if len(s) != len(t):
            return False
        table = {}
        for i in range(len(s)):
            if s[i] not in table.keys():
                table[s[i]] = 1
            else:
                table[s[i]] += 1

        for i in range(len(t)):
            if t[i] not in table.keys():
                return False
            else:
                table[t[i]] -= 1
                if table[t[i]] < 0:
                    return False
        return True
```

## 257-二叉树的所有路径

给定一个二叉树，返回所有从根节点到叶子节点的路径。
说明: 叶子节点是指没有子节点的节点。

示例:
```
输入:

   1
 /   \
2     3
 \
  5

输出: ["1->2->5", "1->3"]

解释: 所有根节点到叶子节点的路径为: 1->2->5, 1->3
```

迭代：
```python
class Solution:
    def binaryTreePaths(self, root):
        def construct_paths(root, path):
            if root:
                path += str(root.val)
                if not root.left and not root.right:  # 当前节点是叶子节点
                    paths.append(path)  # 把路径加入到答案中
                else:
                    path += '->'  # 当前节点不是叶子节点，继续递归遍历
                    construct_paths(root.left, path)
                    construct_paths(root.right, path)

        paths = []
        construct_paths(root, '')
        return paths

作者：LeetCode
链接：https://leetcode-cn.com/problems/binary-tree-paths/solution/er-cha-shu-de-suo-you-lu-jing-by-leetcode/
```

递归：
```python
class Solution:
    def binaryTreePaths(self, root):
        if not root:
            return []

        paths = []
        stack = [(root, str(root.val))]
        while stack:
            node, path = stack.pop()
            if not node.left and not node.right:
                paths.append(path)
            if node.left:
                stack.append((node.left, path + '->' + str(node.left.val)))
            if node.right:
                stack.append((node.right, path + '->' + str(node.right.val)))

        return paths

作者：LeetCode
链接：https://leetcode-cn.com/problems/binary-tree-paths/solution/er-cha-shu-de-suo-you-lu-jing-by-leetcode/
```

## 258-各位相加
给定一个非负整数 num，反复将各个位上的数字相加，直到结果为一位数。

示例:
```
输入: 38
输出: 2
解释: 各位相加的过程为：3 + 8 = 11, 1 + 1 = 2。 由于 2 是一位数，所以返回 2。
```

```python
class Solution:
    def addDigits(self, num: int) -> int:
        return num % 9 or 9 * bool(num)
```

## 268-缺失数字
给定一个包含 0, 1, 2, ..., n 中 n 个数的序列，找出 0 .. n 中没有出现在序列中的那个数。

```python
class Solution:
    def missingNumber(self, nums: List[int]) -> int:
        num_set = set(nums)
        n = len(nums) + 1
        for number in range(n):
            if number not in num_set:
                return number
```
## 283-移动零

给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。

示例:
```
输入: [0,1,0,3,12]
输出: [1,3,12,0,0]
```

```python
class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                nums[j] = nums[i]
                j += 1
        for i in range(j, len(nums)):
            nums[i] = 0
        return nums
```
