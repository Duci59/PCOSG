"""
frontier.py - Định nghĩa FrontierQueue cho Pattern-Aware SOPG
Mục đích:
    - Quản lý frontier trong tree search
    - Ưu tiên các Node có log-prob cao
    - Tránh duplicate prefix đã sinh
"""

import heapq
from typing import List, Set
from .node import Node


class FrontierQueue:
    """
    Hàng đợi ưu tiên (priority queue) để lưu trữ các Node trong search.
    Ưu tiên node có log-prob cao hơn.
    Tránh duplicate thông qua visited set.
    """
    def __init__(self):
        # Priority queue cài bằng heapq
        self.heap: List = []
        # Lưu set các prefix (tuple) đã sinh
        self.visited: Set = set()

    def add(self, node: Node):
        # Thêm node mới vào frontier nếu chưa duplicate.
        # Kiểm tra duplicate bằng prefix_tokens
        prefix_tuple = tuple(node.prefix_tokens)
        if prefix_tuple in self.visited:
            return

        # Đánh dấu đã visit
        self.visited.add(prefix_tuple)

        # Push vào heap (min-heap → đảo dấu)
        heapq.heappush(self.heap, (-node.log_prob, node))

    def pop(self) -> Node:
        # Lấy node có log-prob cao nhất ra khỏi frontier.
        if not self.heap:
            raise IndexError("FrontierQueue is empty!")
        _, node = heapq.heappop(self.heap)
        return node

    def is_empty(self) -> bool:
        # Frontier còn node không.
        return len(self.heap) == 0

    def size(self) -> int:
        # Số lượng node trong frontier.
        return len(self.heap)