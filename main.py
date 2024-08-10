from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from plyer import filechooser
from kivy.clock import mainthread


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

        self.video_path = None
        self.dataframe = None

        return layout

    def select_video(self, instance):
        filechooser.open_file(on_selection=self.handle_selection, filters=[("Video Files", "*.mp4;*.avi;*.mov")])

    @mainthread
    def handle_selection(self, selection):
        if selection:
            self.video_path = selection[0]
            # Process the video
            self.process_video()
        else:
            print("No file selected.")

    def process_video(self):
        # Placeholder for video processing
        # Replace this with your actual video processing function
        import pandas as pd
        self.dataframe = pd.DataFrame({'Sample': [1, 2, 3], 'Data': [4, 5, 6]})
        print("Video processed successfully.")
        self.download_button.disabled = False

    def download_csv(self, instance):
        if self.dataframe is not None:
            filechooser.save_file(on_selection=self.handle_save)

    @mainthread
    def handle_save(self, selection):
        if selection:
            save_path = selection[0]
            if not save_path.endswith('.csv'):
                save_path += '.csv'
            self.dataframe.to_csv(save_path, index=False)
            print(f"CSV saved to {save_path}")
        else:
            print("Save cancelled.")

if __name__ == '__main__':
    VideoProcessorApp().run()