

# Linked List

- [两数相加](#两数相加)
- [删除链表中倒数第N个结点](#删除链表中倒数第N个结点)
- [合并两个有序链表](#合并两个有序链表)
- [合并K个排序链表](#合并K个排序链表)
- [两两交换链表中的节点](#两两交换链表中的节点)
- [旋转链表](#旋转链表)


## 两数相加

```python
class Solution:
    def addTwoNumbers(self, l1: ListNode, l2: ListNode) -> ListNode:
        newNode = ListNode(0)
        ansNode = newNode

        flag = 0
        while l1.val >= 0 or l2.val >=0:
            x = l1.val if l1.val >= 0 else 0
            y = l2.val if l2.val >= 0 else 0
            sum = x + y + flag
            flag = sum // 10
            newNode.val = sum % 10

            l1 = l1.next if l1.next else ListNode(-1)
            l2 = l2.next if l2.next else ListNode(-1)
            if l1.val >= 0 or l2.val >= 0 or flag:
                newNode.next = ListNode(0)
                newNode = newNode.next
        if flag == 1:
            newNode.val = 1
        return ansNode
```

## 删除链表中倒数第N个结点

```python
class Solution:
    def removeNthFromEnd(self, head: ListNode, n: int) -> ListNode:
        node = ListNode(-1)
        node.next = head
        p = q = node
        for i in range(n):
            p = p.next
        while p.next:
            p = p.next
            q = q.next
        q.next = q.next.next
        return node.next
```

## 合并两个有序链表

```python
class Solution:
    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if l1 is None:
            return l2
        elif l2 is None:
            return l1
        elif l1.val >= l2.val:
            l2.next = self.mergeTwoLists(l2.next, l1)
            return l2
        else:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
```

## 合并K个排序链表

```python
class Solution:
    def mergeKLists(self, lists: List[ListNode]) -> ListNode:
        list_all = []
        for l in lists:
            while l:
                list_all.append(l.val)
                l = l.next
        out = point = ListNode(0)
        print(sorted(list_all))
        for i in sorted(list_all):
            point.next = ListNode(i)
            point = point.next
        return out.next
```

## 两两交换链表中的节点

```python
class Solution:
    def swapPairs(self, head: ListNode) -> ListNode:
        if not head or not head.next:
            return head

        first_node = head
        second_node = head.next

        first_node.next = self.swapPairs(second_node.next)
        second_node.next = first_node

        return second_node
```

## 旋转链表

```python
class Solution:
    def rotateRight(self, head: ListNode, k: int) -> ListNode:
        if not head:
            return None
        if not head.next:
            return head

        n = 1
        pre = head
        while pre.next:
            pre = pre.next
            n += 1
        pre.next = head

        new_tail = head
        for i in range(n - k%n -1):
            new_tail = new_tail.next
        new_head = new_tail.next
        new_tail.next = None
        return new_head
```
