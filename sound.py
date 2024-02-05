import pygame

def play_audio(file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():  # wait for the music to finish playing
        pygame.time.Clock().tick(10)  # delay for 10 milliseconds

play_audio('buzzer.mp3')