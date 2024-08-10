from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button

class VideoProcessorApp(App):
    def build(self):
        self.title = 'Video Processor'
        layout = BoxLayout(orientation='vertical', padding=10, spacing=10)

        self.select_button = Button(text='Select Video', size_hint=(1, 0.2))
        self.select_button.bind(on_press=self.select_video)
        layout.add_widget(self.select_button)

        self.download_button = Button(text='Download CSV', size_hint=(1, 0.2), disabled=True)
        self.download_button.bind(on_press=self.download_csv)
        layout.add_widget(self.download_button)

        return layout

    def select_video(self, instance):
        # To be implemented
        pass

    def download_csv(self, instance):
        # To be implemented
        pass

if __name__ == '__main__':
    VideoProcessorApp().run()

    