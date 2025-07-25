from playsound3 import playsound
import threading

def reproducir_audio_menu(sonido, index=1):
    def _play(file):
        threading.Thread(target=lambda: playsound(file)).start()

    if sonido == "correct":
        _play("audios/correct.mp3")
    elif sonido == "selected":
        _play("audios/selected.mp3")
    elif sonido == "serie":
        _play("audios/serie.mp3")
    elif sonido == "start":
        _play("audios/start.mp3")
    elif sonido == "back":
        _play("audios/back.mp3")
    elif sonido == "back_on":
        _play("audios/back_on.mp3")
    elif sonido == "select":
        if index == 1:
            _play("audios/select_1.mp3")
            return 2
        elif index == 2:
            _play("audios/select_2.mp3")
            return 1

    return index  # <- Asegura que siempre devuelve algo

def reproducir_audio_poses(sonido):
    def _play(file):
        threading.Thread(target=lambda: playsound(file)).start()

    if sonido == "correct":
        _play("audios/correct.mp3")
    elif sonido == "selected":
        _play("audios/selected.mp3")
    elif sonido == "serie":
        _play("audios/serie.mp3")
    elif sonido == "start":
        _play("audios/start.mp3")
    elif sonido == "victory":
        _play("audios/victory.mp3")