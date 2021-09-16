from enum import IntEnum


class TTSActions(IntEnum):
    READY = 0
    RESPONSE = 1
    GENERATE = 2
    CHANGE_SPEAKER = 3
    CHANGE_GENERATE = 4
    ERROR = 5


class TTSMode(IntEnum):
    PRECISE = 0
    FAST = 1
