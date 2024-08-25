from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from plyer import filechooser
from kivy.clock import mainthread
from kivy.uix.scrollview import ScrollView
from kivy.config import Config


class CalloutLabel(AnchorLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.anchor_x = "center"
        self.anchor_y = "center"

        callout_label = Label(text="Your weekly change is -0.74kg", size_hint = (0.9, 0.9))
        self.add_widget(callout_label)

class TableGraphSelector(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "horizontal"
        self.padding = 0
        self.spacing = 0

        table_button = Button(text="Table", size_hint = (0.5, 1))
        graph_button = Button(text="Graph", size_hint = (0.5, 1))
        
        self.add_widget(table_button)
        self.add_widget(graph_button)



class WeightScribeApp(App):
    def build(self):
        self.title = 'WeightScribe'
        
        # start with splitting the title bar and app content
        title_body_layout = BoxLayout(orientation='vertical', padding=0, spacing=0)
        title_label = Label(text="WeightScribe", size_hint = (1, 0.1))
        title_body_layout.add_widget(title_label)

        # add the body 
        body_layout = BoxLayout(orientation = "vertical", padding=0, spacing=0, size_hint = (1, 0.8))
        title_body_layout.add_widget(body_layout)

        # body layout split into callout, table/graph selector, table/graph, upload/download

        callout_label = CalloutLabel(size_hint = (1, 0.1))
        body_layout.add_widget(callout_label)

        table_graph_selector = TableGraphSelector(size_hint = (1, 0.05))
        body_layout.add_widget(table_graph_selector)

        table_graph_placeholder = Button(text="table/graph", size_hint = (1, 0.65))
        body_layout.add_widget(table_graph_placeholder)

        upload_download_section = BoxLayout(orientation="horizontal", size_hint = (1, 0.2))

        self.select_button = Button(text='Select Video', size_hint=(0.5, 1))
        self.select_button.bind(on_press=self.select_video)
        upload_download_section.add_widget(self.select_button)

        self.download_button = Button(text='Download CSV', size_hint=(.5, 1), disabled=True)
        self.download_button.bind(on_press=self.download_csv)
        upload_download_section.add_widget(self.download_button)

        body_layout.add_widget(upload_download_section)

        self.video_path = None
        self.dataframe = None

        return title_body_layout

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
    width = 360
    height = int((2000/1080) * width)
    Config.set('graphics', 'width', f'{width}')
    Config.set('graphics', 'height', f'{height}')
    WeightScribeApp().run()