from keras.models import load_model
from tkinter import *
from PIL import ImageGrab
import numpy as np
import win32gui

model = load_model('hwr.h5')
x = y = 0


def predict_digit(img):
    img = img.resize((28, 28))
    img = img.convert('L')
    img = np.array(img).reshape((-1, 28, 28, 1))
    img = img / 255.0
    res = model.predict([img])[0]
    return np.argmax(res), max(res)


def draw(event):
    global x, y

    x = event.x
    y = event.y
    r = 5
    canvas.create_oval(x - r, y - r, x + r, y + r, fill="white", outline="white")


def clearAll():
    canvas.delete("all")
    label.configure(text="Cleared!")


def classify():
    l, t, r, b = win32gui.GetWindowRect(canvas.winfo_id())
    im = ImageGrab.grab(bbox=(l + 10, t + 10, r + 6, b + 6))
    # im.show()
    digit, acc = predict_digit(im)
    label.configure(text="Result: " + str(digit))


screen = Tk()
w, h = screen.winfo_screenwidth(), screen.winfo_screenheight()
screen.attributes('-fullscreen', True)

canvas = Canvas(screen, width=300, height=300, bg="black", cursor="circle")
canvas.grid(row=0, column=0, padx=10, pady=10, sticky=W, columnspan=4)
canvas.bind("<B1-Motion>", draw)

button_clear = Button(screen, text="Clear", command=clearAll)
button_clear.grid(row=1, column=0, pady=2)

classify_btn = Button(screen, text="Classify", command=classify)
classify_btn.grid(row=1, column=1, padx=2, pady=2)

classify_btn = Button(screen, text="Close", command=screen.destroy)
classify_btn.grid(row=1, column=2, padx=2, pady=2)

label = Label(screen, text="No Data!", font=("Helvetica", 14))
label.grid(row=1, column=3, padx=2, pady=2)

mainloop()
