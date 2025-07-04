"""
packing.py - Hỗ trợ packing cho SOPG Frontier

Mục đích:
    - Gom các node có log-prob thấp thành 1 meta-node (PackedNode)
    - Giảm số lượng node trong frontier → tiết kiệm bộ nhớ
"""

from typing import List, Union
from .node import Node


class PackedNode:
    """
    Node đặc biệt đại diện cho nhóm các node con có log-prob thấp.
    Khi cần mở rộng → unpack ra.
    """
    def __init__(self, children: List[Node]):
        self.children = children
        # log_prob đại diện = trung bình log_prob (hoặc min)
        self.log_prob = min(child.log_prob for child in children)

    def __lt__(self, other):
        """
        Hỗ trợ so sánh trong PriorityQueue.
        """
        return self.log_prob > other.log_prob

    def __repr__(self):
        return f"PackedNode(num_children={len(self.children)}, log_prob={self.log_prob:.4f})"


def pack_nodes(nodes: List[Node], threshold: float) -> List[Union[Node, PackedNode]]:
    """
    Gom các node có log_prob thấp hơn threshold vào PackedNode.

    :param nodes: Danh sách Node ban đầu
    :param threshold: Ngưỡng log_prob để quyết định packing
    :return: List Node hoặc PackedNode
    """
    high_prob_nodes = []
    low_prob_nodes = []

    for node in nodes:
        if node.log_prob >= threshold:
            high_prob_nodes.append(node)
        else:
            low_prob_nodes.append(node)

    result = high_prob_nodes

    if low_prob_nodes:
        packed = PackedNode(low_prob_nodes)
        result.append(packed)

    return result


def unpack_node(packed_node: PackedNode) -> List[Node]:
    """
    Giải nén PackedNode thành danh sách node con.
    """
    return packed_node.children
