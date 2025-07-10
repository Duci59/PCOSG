"""
node.py - Định nghĩa lớp Node cho Pattern-Aware SOPG

Mục đích:
    Đại diện một bước trong search tree của quá trình sinh mật khẩu
    - Giữ thông tin prefix đã sinh
    - Tích lũy log-prob
    - Theo dõi trạng thái pattern (L/N/S còn lại)
    - Kiểm tra hợp lệ theo pattern
"""

from typing import List, Dict


class Node:
    """
    Node trong search tree:
    - prefix_tokens: list[int] các token đã sinh
    - log_prob: tổng log-xác suất
    - pattern_state: dict {'L': int, 'N': int, 'S': int} - số ký tự còn lại
    """

    def __init__(
        self,
        prefix_tokens: List[int], 
        log_prob: float,
        depth: int,
        pattern_state: Dict[str, int],
        tokenizer,
        pattern: str
    ):
        """
        Khởi tạo Node:
        prefix_tokens: Chuỗi token đã sinh đến hiện tại
        log_prob: Tổng logarit xác suất
        depth: Vị trí hiện tại trong pattern
        pattern_state: dict {'L': remaining, 'N': remaining, 'S': remaining} kiểm tra số lượng ký tự còn lại để tạo
        tokenizer: Đối tượng tokenizer
        pattern: Pattern gốc
        """
        self.prefix_tokens = prefix_tokens
        self.log_prob = log_prob
        self.depth = depth
        self.pattern_state = pattern_state
        self.tokenizer = tokenizer
        self.pattern = pattern

    def is_terminal(self) -> bool:
        # Kiểm tra node đã hoàn thành pattern chưa.
        return all(v == 0 for v in self.pattern_state.values())

    def allowed_token_types(self) -> List[str]:
        # Trả về danh sách các loại ký tự còn được phép sinh.
        return [k for k, v in self.pattern_state.items() if v > 0]

    def update_pattern_state(self, next_char_type: str) -> Dict[str, int]:
        # Trả về bản sao mới của pattern_state sau khi sinh 1 ký tự. Giảm count tương ứng với loại ký tự.

        # next_char_type: 'L', 'N', hoặc 'S'
        new_state = self.pattern_state.copy()
        if next_char_type in new_state and new_state[next_char_type] > 0:
            new_state[next_char_type] -= 1

        # Trả về dict mới với số đếm cập nhật
        return new_state

    def copy_with_new_token(self, new_token: int, new_log_prob: float) -> 'Node':
        # Tạo node mới với prefix mở rộng thêm 1 token.

        # new_token: token ID mới được sinh
        # new_log_prob: log-prob của token mới

        # Xác định loại ký tự của token mới
        next_char_type = self.tokenizer.get_char_type(new_token)

        # Nếu loại ký tự này không hợp lệ theo pattern → raise error
        if next_char_type not in self.pattern_state:
            raise ValueError(f"Token type {next_char_type} không tồn tại trong pattern_state!")

        # Cập nhật pattern_state
        new_pattern_state = self.update_pattern_state(next_char_type)

        # Trả về Node mới
        return Node(
            prefix_tokens=self.prefix_tokens + [new_token],
            log_prob=self.log_prob + new_log_prob,
            depth=self.depth + 1,
            pattern_state=new_pattern_state,
            tokenizer=self.tokenizer,
            pattern=self.pattern
        )

    def __lt__(self, other: 'Node') -> bool:
        """
        Hàm so sánh để tích hợp với PriorityQueue.
        Node có log_prob cao hơn sẽ được ưu tiên.
        (PriorityQueue trong Python là min-heap → đảo dấu so sánh)
        """
        return self.log_prob > other.log_prob

    def __repr__(self) -> str:
        return (
            f"Node(prefix_tokens={self.prefix_tokens}, "
            f"log_prob={self.log_prob:.4f}, "
            f"pattern_state={self.pattern_state})"
        )
