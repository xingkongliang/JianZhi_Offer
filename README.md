# 剑指Offer


## 二维数组中的查找

题目描述:

在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
```
链接：https://www.nowcoder.com/questionTerminal/abc3fe2ce8e146608e868a70efebf62e?f=discussion
来源：牛客网

class Solution:
    # array 二维列表
    def Find(self, target, array):
        # write code here
        rows = len(array) - 1
        cols= len(array[0]) - 1
        i = rows
        j = 0
        while j<=cols and i>=0:
            if target<array[i][j]:
                i -= 1
            elif target>array[i][j]:
                j += 1
            else:
                return True
        return False
```


## 把数组排成最小的数

题目描述:

输入一个正整数数组，把数组里所有数字拼接起来排成一个数，打印能拼接出的所有数字中最小的一个。例如输入数组{3，32，321}，则打印出这三个数字能排成的最小数字为321323。

```python
链接：https://www.nowcoder.com/questionTerminal/8fecd3f8ba334add803bf2a06af1b993?f=discussion
来源：牛客网

class Solution:
    def PrintMinNumber(self, numbers):
        # write code here
        if not numbers: return ""
        numbers = list(map(str, numbers))
        numbers.sort(cmp=lambda x, y: cmp(x + y, y + x))
        return "".join(numbers).lstrip('0') or'0'
```

```python
链接：https://www.nowcoder.com/questionTerminal/8fecd3f8ba334add803bf2a06af1b993?f=discussion
来源：牛客网

class Solution:
    def compare(self,num1,num2):
        t = str(num1)+str(num2)
        s = str(num2)+str(num1)
        if t>s:
            return 1
        elif t<s:
            return -1
        else:
            return 0
 
    def PrintMinNumber(self, numbers):
        # write code here
        if numbers is None:
            return ""
        lens = len(numbers)
        if lens ==0 :
            return ""
        tmpNumbers = sorted(numbers,cmp=self.compare)
        return int(''.join(str(x)for x in tmpNumbers))
 
print Solution().PrintMinNumber([3,32,321])
```
