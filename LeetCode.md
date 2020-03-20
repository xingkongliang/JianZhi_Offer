


   * 面试题：[通配符匹配|greedy|](#通配符匹配)
   * 面试题：[跳跃游戏II|greedy|](#跳跃游戏II)
   * 面试题：[加油站](#加油站|greedy|)
   * 面试题：[135-分发糖果|greedy|](#135-分发糖果)
   * 面试题：[TEMP](#TEMP)
   * 面试题：[TEMP](#TEMP)
   * 面试题：[TEMP](#TEMP)
   * 面试题：[TEMP](#TEMP)
   * 面试题：[TEMP](#TEMP)

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
