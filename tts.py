import os
import time
from redis import Sentinel, Redis
import json
import base64

from tts_enum import TTSActions, TTSMode
from logger import logger
from typing import Any, Dict, List, Tuple

from buffer import buffer
from hifi import TaHifiTTS, DEFAULT_GATE_THRESHOLD
from fast_hifi import DEFAULT_PACE, DEFAULT_PITCH_SHIFT, FastHifiTTS


def worker(wid: str, redis_username: str, redis_password: str, redis_master_name: str, addrsses: List[Tuple[str, int]], database: int, sentinel: bool, set_key: str):
    models: Dict[str, TaHifiTTS] = {}
    models_fast: Dict[str, FastHifiTTS] = {}

    if sentinel:
        r: Redis = Sentinel(addrsses, socket_timeout=0.1, sentinel_kwargs={'password': redis_password, 'username': redis_username}).master_for(redis_master_name, password=redis_password, username=redis_username, db=database)
    else:
        r: Redis = Redis(host=addrsses[0][0], port=addrsses[0][1], db=database, username=redis_username, password=redis_password)
    while True:
        try:
            data = r.spop(set_key)
            if not data:
                time.sleep(0.1)
                continue

            event_data: Dict[str, Any] = json.loads(data)

            jid = event_data["jid"]
            event = event_data["event"]
            response_event = event_data["response_event"]
            payload: Dict[str, Any] = event_data["payload"]
            start = time.time()
            mode = event_data.get("mode")

            speaker = payload.get("speaker")

            try:
                if event == TTSActions.CHANGE_GENERATE:
                    if mode == TTSMode.PRECISE:
                        if speaker not in models:
                            models[speaker] = TaHifiTTS()
                            models[speaker].update_model(
                                taco_path=payload.get("taco_path"),
                                onnx_path=payload.get("onnx_path"),
                                speaker=payload.get("speaker"),
                                gpu=payload.get("gpu", False),
                                period=payload.get("period", False),
                                warm_up=payload.get("warm_up", False),
                                gate_threshold=payload.get(
                                    "gate_threshold", DEFAULT_GATE_THRESHOLD
                                ),
                            )
                            logger.info(f"speaker changed to {models[speaker].get_speaker()}")
                        _md = models[speaker]
                    elif mode == TTSMode.FAST:
                        if speaker not in models_fast:
                            models_fast[speaker] = FastHifiTTS()
                            models_fast[speaker].update_model(
                                fast_path=payload.get("fast_path"),
                                onnx_path=payload.get("onnx_path"),
                                cmudict_path=payload.get("cmudict_path"),
                                speaker=payload.get("speaker"),
                                warm_up=payload.get("warm_up", False),
                                gpu=payload.get("gpu", False),
                                pace=payload.get("pace", DEFAULT_PACE),
                                pitch_shift=payload.get(
                                    "pitch_shift", DEFAULT_PITCH_SHIFT
                                ),
                                p_arpabet=payload.get("p_arpabet", 1),
                                period=payload.get("period", False),
                                start=payload.get("start", 0),
                                volume=payload.get("volume", 1),
                                energy_conditioning=payload.get("energy", False),
                            )
                            logger.info(
                                f"[FAST] speaker changed to {models_fast[speaker].get_speaker()}"
                            )
                        _md = models_fast[speaker]
                    else:
                        raise Exception(f"unknown mode: {mode}")
                    with buffer() as out:
                        length = _md.generate(out=out, text=payload.get("text"))

                        data = base64.b64encode(out.getvalue()).decode("ascii")

                        r.publish(
                            response_event,
                            json.dumps(
                                {
                                    "event": TTSActions.RESPONSE,
                                    "jid": jid,
                                    "wid": wid,
                                    "payload": {
                                        "data": data,
                                        "length": length,
                                        "speaker": _md.get_speaker(),
                                    },
                                    "content_length": len(data),
                                    "time": time.time() - start,
                                }
                            ),
                        )
                else:
                    raise Exception(f"unknown event: {event}")
            except Exception as e:
                r.publish(
                    response_event,
                    json.dumps(
                        {
                            "event": TTSActions.ERROR,
                            "jid": jid,
                            "wid": wid,
                            "payload": {
                                "error": str(e),
                            },
                            "time": time.time() - start,
                            "request": payload,
                        }
                    ),
                )
                raise

            logger.info(f"completed req {jid}")
        except KeyboardInterrupt:
            break
        except Exception:
            logger.error(
                "An exception occurred in a tahifi instance, deadlock is now possible",
                exc_info=True,
            )
            continue


if __name__ == "__main__":

    def init():
        logger.info("Starting")

        wid = os.getenv("YAPPER_WORKER_ID")
        redis_uri = os.getenv("YAPPER_REDIS_URI")
        set_key = os.getenv("YAPPER_SET_KEY")

        worker(
            wid=wid,
            redis_uri=redis_uri,
            set_key=set_key,
        )

    init()
