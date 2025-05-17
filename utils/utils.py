import win32gui

def osu_pixels_to_normal_coords(osu_x, osu_y, resolution_width, resolution_height):
    OSU_WIDTH = 512
    OSU_HEIGHT = 384
    playfield_height = 0.8 * resolution_height
    playfield_width = (OSU_WIDTH / OSU_HEIGHT) * playfield_height
    playfield_left = (resolution_width - playfield_width) / 2
    playfield_top = (resolution_height - playfield_height) / 2
    scale = playfield_height / OSU_HEIGHT
    screen_x = playfield_left + (osu_x * scale)
    screen_y = playfield_top + (osu_y * scale)
    return screen_x, screen_y

def get_active_window_name():
    return win32gui.GetWindowText(win32gui.GetForegroundWindow())