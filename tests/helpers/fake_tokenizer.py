class FakeTokenizer:
    def __init__(self, pieces=None):
        self.pieces = dict(pieces or {})
        self.bos_token_id = 0
        self.eos_token_id = 2
        self.pad_token_id = 1
        self.unk_token_id = 3
        self.is_fast = False

    def encode(self, text, add_special_tokens=False):
        if add_special_tokens:
            raise ValueError("FakeTokenizer only supports add_special_tokens=False")
        if text in self.pieces:
            return list(self.pieces[text])
        return [ord(char) for char in text]
