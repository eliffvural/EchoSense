from kivy.app import App
from kivy.uix.button import Button

class MyApp(App):
    def build(self):
        return Button(text="Merhaba Dünya!", font_size=24)

if __name__ == "__main__":
    MyApp().run()
