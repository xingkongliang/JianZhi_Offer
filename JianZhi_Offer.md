# 剑指Offer

* **剑指Offer题解**
   * 面试题3：[二维数组中的查找](#数组中重复的数字)
   * 面试题4：[替换空格](#替换空格)
   * 面试题5：[从尾到头打印链表](#从尾到头打印链表)
   * 面试题6：[重建二叉树](#重建二叉树)
   * 面试题7：[用两个栈实现队列](#用两个栈实现队列)
   * 面试题8：[旋转数组中最小数字](#旋转数组中最小数字)
   * 面试题9：[斐波那契数列](#斐波那契数列)
   * 面试题9-2：[跳台阶](#跳台阶)
   * 面试题9-3：[变态跳台阶](#变态跳台阶)
   * 面试题9-4：[矩形覆盖](#矩形覆盖)
   * 面试题10：[二进制中1的个数](#二进制中1的个数)
   * 面试题11：[数值的整数次方](#数值的整数次方)
   * 面试题12：[打印1到最大的n位数](#打印1到最大的n位数)
   * 面试题13：[在O_1时间删除链表结点](#在O_1时间删除链表结点)
   * 面试题14：[调整数组顺序使奇数位于偶数前面](#调整数组顺序使奇数位于偶数前面)
   * 面试题15：[链表中倒数第k个结点](#链表中倒数第k个结点)
   * 面试题16：[反转链表](#反转链表)
   * 面试题17：[合并两个排序的链表](#合并两个排序的链表)
   * 面试题18：[数的子结构](#数的子结构)
   * 面试题19：[二叉树的镜像](#二叉树的镜像)
   * 面试题20：[顺时针打印矩阵](#顺时针打印矩阵)

   * 面试题：[TEMP](#TEMP)
   * 面试题：[TEMP](#TEMP)


   * 面试题：[把数组排成最小的数](#把数组排成最小的数)
   * 面试题：[二叉搜索树的后续遍历序列](#二叉搜索树的后续遍历序列)
   * 面试题：[从上往下打印二叉树](#从上往下打印二叉树)
   * 面试题：[删除链表中重复的结点](#删除链表中重复的结点)
   * 面试题：[复杂链表的复制](#复杂链表的复制)
   * 面试题37：[两个链表的第一个公共结点](#两个链表的第一个公共结点)
   * 面试题38：[数字在排序数组中出现的次数](#数字在排序数组中出现的次数)
   * 面试题39：[二叉树的深度](#二叉树的深度)
   * 面试题39-2：[平衡二叉树](#平衡二叉树)
   * 面试题40：[数组中只出现一次的数字](#数组中只出现一次的数字)
   * 面试题：[TEMP](#TEMP)


## 二维数组中的查找

题目描述：
在一个二维数组中（每个一维数组的长度相同），每一行都按照从左到右递增的顺序排序，每一列都按照从上到下递增的顺序排序。请完成一个函数，输入这样的一个二维数组和一个整数，判断数组中是否含有该整数。
```python
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

## 替换空格
题目描述：
请实现一个函数，将一个字符串中的每个空格替换成“%20”。例如，当字符串为We Are Happy.则经过替换之后的字符串为We%20Are%20Happy。
```python
class Solution:
    def replaceSpace(self, s):
        ans = []
        for i in s:
            if i == ' ':
                ans.append('%20')
            else:
                ans.append(i)
        return "".join(ans)
```


## 从尾到头打印链表
题目描述：
输入一个链表，按链表从尾到头的顺序返回一个ArrayList。
```python
链接：https://www.nowcoder.com/questionTerminal/d0267f7f55b3412ba93bd35cfa8e8035?f=discussion
来源：牛客网

class Solution:
    # 返回从尾部到头部的列表值序列，例如[1,2,3]
    def printListFromTailToHead(self, listNode):
        # write code here
        lst,lst_bak = [],[]
        if not listNode:
            return lst
        while listNode:
            lst.append(listNode.val)
            listNode = listNode.next
        while lst:
            lst_bak.append(lst.pop())
        return lst_bak
```


## 重建二叉树

题目描述：
输入某二叉树的前序遍历和中序遍历的结果，请重建出该二叉树。假设输入的前序遍历和中序遍历的结果中都不含重复的数字。例如输入前序遍历序列{1,2,4,7,3,5,6,8}和中序遍历序列{4,7,2,1,5,3,8,6}，则重建二叉树并返回。
```python
链接：https://www.nowcoder.com/questionTerminal/8a19cbe657394eeaac2f6ea9b0f6fcf6?f=discussion
来源：牛客网

class Solution:
    # 返回构造的TreeNode根节点
    def reConstructBinaryTree(self, pre, tin):
        # write code here
        if len(pre) == 0:
            return None
        elif len(pre) == 1:
            return TreeNode(pre[0])
        else:
            ans = TreeNode(pre[0])
            ans.left = self.reConstructBinaryTree(pre[1:tin.index(pre[0])+1], tin[:tin.index(pre[0])])
            ans.right = self.reConstructBinaryTree(pre[tin.index(pre[0])+1:], tin[tin.index(pre[0])+1:])
            return ans
```

## 用两个栈实现队列

题目描述：
用两个栈来实现一个队列，完成队列的Push和Pop操作。 队列中的元素为int类型。
```python
链接：https://www.nowcoder.com/questionTerminal/54275ddae22f475981afa2244dd448c6?f=discussion
来源：牛客网

class Solution:
    def __init__(self):
        self.stackA = []
        self.stackB = []

    def push(self, node):
        # write code here
        self.stackA.append(node)

    def pop(self):
        # return xx
        if self.stackB:
            return self.stackB.pop()
        elif not self.stackA:
            return None
        else:
            while self.stackA:
                self.stackB.append(self.stackA.pop())
            return self.stackB.pop()
```

## 旋转数组中最小数字

题目描述：
把一个数组最开始的若干个元素搬到数组的末尾，我们称之为数组的旋转。
输入一个非递减排序的数组的一个旋转，输出旋转数组的最小元素。
例如数组{3,4,5,1,2}为{1,2,3,4,5}的一个旋转，该数组的最小值为1。
NOTE：给出的所有元素都大于0，若数组大小为0，请返回0。

```python
class Solution:
    def minNumberInRotateArray(self, rotateArray):
        if len(rotateArray) == 1:
            return rotateArray[0]
        right = len(rotateArray) - 1
        left = 0

        while left < right:
            if right - left == 1:
                mid = right
                break
            mid = (left + right) // 2
            if rotateArray[mid] >= rotateArray[left]:
                left = mid
            elif rotateArray[mid] <= rotateArray[right]:
                right = mid
        return rotateArray[mid]
```

## 斐波那契数列

题目描述：
大家都知道斐波那契数列，现在要求输入一个整数n，请你输出斐波那契数列的第n项（从0开始，第0项为0）。
n<=39

```python
class Solution:
    def Fibonacci(self, n):
        # write code here
        result = [0, 1]
        if n in result:
            return result[n]
        fibNMinusOne = 1
        fibNMinusTwo = 0
        for i in range(2, n+1):
            fibN = fibNMinusOne + fibNMinusTwo

            fibNMinusTwo = fibNMinusOne
            fibNMinusOne = fibN
        return fibN
```

## 跳台阶

题目描述：
一只青蛙一次可以跳上1级台阶，也可以跳上2级。求该青蛙跳上一个n级的台阶总共有多少种跳法（先后次序不同算不同的结果）。

```python
class Solution:
    def jumpFloor(self, number):
        res = [1, 2]
        if number in res:
            return number

        fibNMinusTwo = 1
        fibNMinusOne = 2
        for i in range(2, number):
            fibN = fibNMinusTwo + fibNMinusOne
            fibNMinusTwo = fibNMinusOne
            fibNMinusOne = fibN
        return fibN
```

## 变态跳台阶

题目描述
一只青蛙一次可以跳上1级台阶，也可以跳上2级……它也可以跳上n级。求该青蛙跳上一个n级的台阶总共有多少种跳法。

```python
class Solution:
    def jumpFloorII(self, number):
        return 2**(number-1)
```

## 矩形覆盖
题目描述：
我们可以用2*1的小矩形横着或者竖着去覆盖更大的矩形。请问用n个2*1的小矩形无重叠地覆盖一个2*n的大矩形，总共有多少种方法？

比如n=3时，2*3的矩形块有3种覆盖方法：

```python
class Solution:
    def rectCover(self, number):
        res = [0, 1, 2]
        if number in res:
            return number

        fibNMinusTwo = 1
        fibNMinusOne = 2
        for i in range(2, number):
            fibN = fibNMinusTwo + fibNMinusOne
            fibNMinusTwo = fibNMinusOne
            fibNMinusOne = fibN
        return fibN
```

## 二进制中1的个数
题目描述：
输入一个整数，输出该数二进制表示中1的个数。其中负数用补码表示。

```python
class Solution:
    def NumberOf1(self, n):
        count = 0
        if n < 0:
            n = n & 0xffffffff
        while n:
            count += 1
            n = (n - 1) & n
        return count
```

## 数值的整数次方
题目描述：
给定一个double类型的浮点数base和int类型的整数exponent。求base的exponent次方。

保证base和exponent不同时为0

```python
class Solution:
    def Power(self, base, exponent):
        abs_exponent = abs(exponent)
        ans = base
        for i in range(abs_exponent-1):
            ans *= base
        if exponent < 0:
            return 1/ans
        elif exponent == 0 and base != 0:
            return 1
        elif exponent == 0 and base == 0:
            return 0
        else:
            return ans
```

## 打印1到最大的n位数
题目描述：
输入数字n，按顺序打印出从1最大的n位十进制数。比如输入3，则打印出1、2、3一直到最大的3位数即999。

```python

```

## 在O_1时间删除链表结点
题目描述：
给定单向链表的头指针和一个结点指针，定义一个函数在O(1)时间删除该结点。

```python

```

## 链表中倒数第k个结点

题目描述：
输入一个链表，输出该链表中倒数第k个结点。

```python
class Solution:
    def FindKthToTail(self, head, k):
        if head == None or k <= 0:
            return None
        p1 = head
        p2 = head
        for i in range(k):
            if p1:
                p1 = p1.next
            else:
                return None
        while p1:
            p1 = p1.next
            p2 = p2.next
        return p2
```

## 调整数组顺序使奇数位于偶数前面

题目描述
输入一个整数数组，实现一个函数来调整该数组中数字的顺序，使得所有的奇数位于数组的前半部分，所有的偶数位于数组的后半部分，并保证奇数和奇数，偶数和偶数之间的相对位置不变。

```python
class Solution:
    def reOrderArray(self, array):
        m = len(array)
        k = 0  # 记录已经摆好位置的奇数的个数
        for i in range(m):
            if array[i] % 2 == 1:
                j = i
                while j > k:
                    tmp = array[j]
                    array[j] = array[j-1]
                    array[j-1] = tmp
                    j -= 1
                k += 1
        return array
```

如果题目中不要求保持相对位置不变，则可使用如下代码：
```python
class Solution:
    def reOrderArray(self, array):
        pBegin = 0
        pEnd = len(array) - 1
        while pBegin < pEnd:
            while pBegin < pEnd and array[pBegin] % 2 != 0:
                pBegin += 1
            while pBegin < pEnd and array[pEnd] % 2 != 1:
                pEnd -= 1
            if pBegin < pEnd:
                temp = array[pBegin]
                array[pBegin] = array[pEnd]
                array[pEnd] = temp
        return array
```

## 反转链表
题目描述：
输入一个链表，反转链表后，输出新链表的表头。
```python
链接：https://www.nowcoder.com/questionTerminal/75e878df47f24fdc9dc3e400ec6058ca?f=discussion
来源：牛客网

class Solution:
    # 返回ListNode
    def ReverseList(self, pHead):
        # write code here
        if pHead==None or pHead.next==None:
            return pHead
        pre = None
        cur = pHead
        while cur!=None:
            tmp = cur.next
            cur.next = pre
            pre = cur
            cur = tmp
        return pre
```

## 合并两个排序的链表
题目描述
输入两个单调递增的链表，输出两个链表合成后的链表，当然我们需要合成后的链表满足单调不减规则。

递归版
```python
链接：https://www.nowcoder.com/questionTerminal/d8b6b4358f774294a89de2a6ac4d9337?f=discussion
来源：牛客网

public ListNode Merge(ListNode list1,ListNode list2) {
       if(list1 == null){
           return list2;
       }
       if(list2 == null){
           return list1;
       }
       if(list1.val <= list2.val){
           list1.next = Merge(list1.next, list2);
           return list1;
       }else{
           list2.next = Merge(list1, list2.next);
           return list2;
       }       
   }
```

非递归版
```python
链接：https://www.nowcoder.com/questionTerminal/d8b6b4358f774294a89de2a6ac4d9337?f=discussion
来源：牛客网

if(list1 == null){
            return list2;
        }
        if(list2 == null){
            return list1;
        }
        ListNode mergeHead = null;
        ListNode current = null;     
        while(list1!=null && list2!=null){
            if(list1.val <= list2.val){
                if(mergeHead == null){
                   mergeHead = current = list1;
                }else{
                   current.next = list1;
                   current = current.next;
                }
                list1 = list1.next;
            }else{
                if(mergeHead == null){
                   mergeHead = current = list2;
                }else{
                   current.next = list2;
                   current = current.next;
                }
                list2 = list2.next;
            }
        }
        if(list1 == null){
            current.next = list2;
        }else{
            current.next = list1;
        }
        return mergeHead;

```

## 数的子结构

题目描述
输入两棵二叉树A，B，判断B是不是A的子结构。（ps：我们约定空树不是任意一个树的子结构）

```python
链接：https://www.nowcoder.com/questionTerminal/6e196c44c7004d15b1610b9afca8bd88?f=discussion
来源：牛客网

class Solution:
    def HasSubtree(self, pRoot1, pRoot2):
        # write code here
        result = False
        if pRoot1 != None and pRoot2 != None:
            if pRoot1.val == pRoot2.val:
                result = self.DoesTree1haveTree2(pRoot1, pRoot2)
            if not result:
                result = self.HasSubtree(pRoot1.left, pRoot2)
            if not result:
                result = self.HasSubtree(pRoot1.right, pRoot2)
        return result
    # 用于递归判断树的每个节点是否相同
    # 需要注意的地方是: 前两个if语句不可以颠倒顺序
    # 如果颠倒顺序, 会先判断pRoot1是否为None, 其实这个时候pRoot2的结点已经遍历完成确定相等了, 但是返回了False, 判断错误
    def DoesTree1haveTree2(self, pRoot1, pRoot2):
        if pRoot2 == None:
            return True
        if pRoot1 == None:
            return False
        if pRoot1.val != pRoot2.val:
            return False
        return self.DoesTree1haveTree2(pRoot1.left, pRoot2.left) and self.DoesTree1haveTree2(pRoot1.right, pRoot2.right)
```
```python
链接：https://www.nowcoder.com/questionTerminal/6e196c44c7004d15b1610b9afca8bd88?f=discussion
来源：牛客网

class Solution:
    def HasSubtree(self, pRoot1, pRoot2):

        def convert(p):
            if p:
                return str(p.val) +  convert(p.left) + convert(p.right)
            else:
                return ""
        return convert(pRoot2) in convert(pRoot1) if pRoot2 else False
```

## 二叉树的镜像

题目描述：
操作给定的二叉树，将其变换为源二叉树的镜像。
输入描述:
二叉树的镜像定义：源二叉树
    	    8
    	   /  \
    	  6   10
    	 / \  / \
    	5  7 9 11
    	镜像二叉树
    	    8
    	   /  \
    	  10   6
    	 / \  / \
    	11 9 7  5

```python
class Solution:
    def Mirror(self, root):
        if not root:
            return None
        if root.left:
            root.left = self.Mirror(root.left)
        if root.right:
            root.right = self.Mirror(root.right)
        temp = root.left
        root.left = root.right
        root.right = temp
        return root
```

## 顺时针打印矩阵
题目描述：
输入一个矩阵，按照从外向里以顺时针的顺序依次打印出每一个数字，例如，如果输入如下4 X 4矩阵： 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 则依次打印出数字1,2,3,4,8,12,16,15,14,13,9,5,6,7,11,10.

```python

```

## 两个链表的第一个公共结点
题目描述
输入两个链表，找出它们的第一个公共结点。（注意因为传入数据是链表，所以错误测试数据的提示是用其他方式显示的，保证传入数据是正确的）
```python
链接：https://www.nowcoder.com/questionTerminal/6ab1d9a29e88450685099d45c9e31e46?f=discussion
来源：牛客网

class Solution:
    def FindFirstCommonNode(self, pHead1, pHead2):
        p1,p2=pHead1,pHead2
        while p1!=p2:
            p1 = p1.next if p1 else pHead2
            p2 = p2.next if p2 else pHead1
        return p1
```

## 复杂链表的复制
题目描述
输入一个复杂链表（每个节点中有节点值，以及两个指针，一个指向下一个节点，另一个特殊指针指向任意一个节点），返回结果为复制后复杂链表的head。（注意，输出结果中请不要返回参数中的节点引用，否则判题程序会直接返回空）

```python
链接：https://www.nowcoder.com/questionTerminal/f836b2c43afc4b35ad6adc41ec941dba?f=discussion
来源：牛客网

class Solution:
    # 返回 RandomListNode
    def Clone(self, pHead):
        if not pHead:
            return None

        dummy = pHead

        # first step, N' to N next
        while dummy:
            dummynext = dummy.next
            copynode = RandomListNode(dummy.label)
            copynode.next = dummynext
            dummy.next = copynode
            dummy = dummynext

        dummy = pHead

        # second step, random' to random'
        while dummy:
            dummyrandom = dummy.random
            copynode = dummy.next
            if dummyrandom:
                copynode.random = dummyrandom.next
            dummy = copynode.next

        # third step, split linked list
        dummy = pHead
        copyHead = pHead.next
        while dummy:
            copyNode = dummy.next
            dummynext = copyNode.next
            dummy.next = dummynext
            if dummynext:
                copyNode.next = dummynext.next
            else:
                copyNode.next = None
            dummy = dummynext

        return copyHead
```

## 二叉搜索树的后续遍历序列

题目描述
输入一个整数数组，判断该数组是不是某二叉搜索树的后序遍历的结果。如果是则输出Yes,否则输出No。假设输入的数组的任意两个数字都互不相同。
```python
链接：https://www.nowcoder.com/questionTerminal/a861533d45854474ac791d90e447bafd?f=discussion
来源：牛客网

class Solution:
    def VerifySquenceOfBST(self, sequence):
        # write code here
        if not sequence:
            return False

        return self.helper(sequence)

    # 增加helper函数是因为对于递归来说sequence为空可以作为终止条件，而对于判断BST而言 sequence为空是False
    def helper(self, sequence):
        if not sequence:
            return True

        root = sequence[-1]
        for i in range(len(sequence)):
            if sequence[i] > root:
                break

        for right in sequence[i:-1]:
            if right < root:
                return False

        return self.helper(sequence[:i]) and self.helper(sequence[i:-1])
```

## 从上往下打印二叉树
题目描述
从上往下打印出二叉树的每个节点，同层节点从左至右打印。

```python
链接：https://www.nowcoder.com/questionTerminal/7fe2212963db4790b57431d9ed259701?f=discussion
来源：牛客网

class Solution:
    # 返回从上到下每个节点值列表，例：[1,2,3]
    def PrintFromTopToBottom(self, root):
        # write code here
        if not root:
            return []
        queue = []
        result = []

        queue.append(root)
        while len(queue) > 0:
            node = queue.pop(0)
            result.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        return result
```

## 删除链表中重复的结点

题目描述
在一个排序的链表中，存在重复的结点，请删除该链表中重复的结点，重复的结点不保留，返回链表头指针。 例如，链表1->2->3->3->4->4->5 处理后为 1->2->5

```python
链接：https://www.nowcoder.com/questionTerminal/fc533c45b73a41b0b44ccba763f866ef?f=discussion
来源：牛客网

class Solution:
    def deleteDuplication(self, pHead):
        # write code here
        if pHead == None or pHead.next == None:
            return pHead
        new_head = ListNode(-1)
        new_head.next = pHead
        pre = new_head
        p = pHead
        nex = None
        while p != None and p.next != None:
            nex = p.next
            if p.val == nex.val:
                while nex != None and nex.val == p.val:
                    nex = nex.next
                pre.next = nex
                p = nex
            else:
                pre = p
                p = p.next
        return new_head.next
```

## 数字在排序数组中出现的次数

题目描述：
统计一个数字在排序数组中出现的次数。

```python
class Solution:
    def GetFirstK(self, data, k):
        low = 0
        high = len(data) - 1
        while low <= high:
            mid = (low + high) // 2
            if data[mid] < k:
                low = mid + 1
            elif data[mid] > k:
                high = mid - 1
            else:
                if mid == low or data[mid - 1] != k: #当到list[0]或不为k的时候跳出函数
                    return mid
                else:
                    high = mid - 1
        return -1

    def GetLastK(self, data, k):
        low = 0
        high = len(data) - 1
        while low <= high:
            mid = (low + high) // 2
            if data[mid] > k:
                high = mid - 1
            elif data[mid] < k:
                low = mid + 1
            else:
                if mid == high or data[mid + 1] != k:
                    return mid
                else:
                    low = mid + 1
        return -1

    def GetNumberOfK(self, data, k):
        if not data:
            return 0
        if self.GetLastK(data, k) == -1 and self.GetFirstK(data, k) == -1:
            return 0
        return self.GetLastK(data, k) - self.GetFirstK(data, k) + 1
```

## 二叉树的深度

题目描述：
输入一棵二叉树，求该树的深度。从根结点到叶结点依次经过的结点（含根、叶结点）形成树的一条路径，最长路径的长度为树的深度。

```python
class Solution:
    def TreeDepth(self, pRoot):
        if pRoot == None:
            return 0
        nLeft = self.TreeDepth(pRoot.left)
        nRight = self.TreeDepth(pRoot.right)
        return (nLeft + 1) if nLeft > nRight else (nRight + 1)
```

## 平衡二叉树

题目描述：
输入一棵二叉树，判断该二叉树是否是平衡二叉树。

```python
class Solution:
    def TreeDepth(self, pRoot):
        if pRoot == None:
            return 0
        nLeft = self.TreeDepth(pRoot.left)
        nRight = self.TreeDepth(pRoot.right)

        return (nLeft + 1) if nLeft > nRight else (nRight + 1)

    def IsBalanced_Solution(self, pRoot):
        if pRoot == None:
            return True
        nLeft = self.TreeDepth(pRoot.left)
        nRight = self.TreeDepth(pRoot.right)
        diff = nLeft - nRight
        if abs(diff) > 1:
            return False
        return self.IsBalanced_Solution(pRoot.left) and self.IsBalanced_Solution(pRoot.right)
```

## 数组中只出现一次的数字

题目描述：
一个整型数组里除了两个数字之外，其他的数字都出现了两次。请写程序找出这两个只出现一次的数字。
```python
链接：https://www.nowcoder.com/questionTerminal/e02fdb54d7524710a7d664d082bb7811?f=discussion
来源：牛客网


# hashMap法
class Solution:
    # 返回[a,b] 其中ab是出现一次的两个数字
    def FindNumsAppearOnce(self, array):
        hashMap = {}
        for i in array:
            if str(i) in hashMap:
                hashMap[str(i)] += 1
            else:
                hashMap[str(i)] = 1
        res = []
        for k in hashMap.keys():
            if hashMap[k] == 1:
                res.append(int(k))
        return res

# 异或法
class Solution:
    def FindNumsAppearOnce(self, array):
        if not array:
            return []
        # 对array中的数字进行异或运算
        tmp = 0
        for i in array:
            tmp ^= i
        # 获取tmp中最低位1的位置
        idx = 0
        while (tmp & 1) == 0:
            tmp >>= 1
            idx += 1
        a = b = 0
        for i in array:
            if self.isBit(i, idx):
                a ^= i
            else:
                b ^= i
        return [a, b]

    def isBit(self, num, idx):
        """
        判断num的二进制从低到高idx位是不是1
        :param num: 数字
        :param idx: 二进制从低到高位置
        :return: num的idx位是否为1
        """
        num = num >> idx
        return num & 1
```
## 把数组排成最小的数

题目描述：
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
