from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.label import Label

class MyApp(App):
    def build(self):
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(Label(text="Merhaba!"))
        layout.add_widget(Button(text="Tıklayın!"))
        return layout

if __name__ == "__main__":
    MyApp().run()
