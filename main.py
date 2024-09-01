from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from plyer import filechooser
from kivy.clock import mainthread
from kivy.uix.scrollview import ScrollView
from kivy.config import Config
import pandas as pd
import datetime as dt
from scipy.stats import linregress
from kivy_garden.graph import Graph, MeshLinePlot
import matplotlib.pyplot as plt


class CalloutLabel(AnchorLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.anchor_x = "center"
        self.anchor_y = "center"

        callout_value = self.get_callout()

        callout_label = Label(text=f"Your weekly change is {callout_value}kg", size_hint = (0.9, 0.9))
        self.add_widget(callout_label)

    def get_callout(self):
        # grab data
        df = App.get_running_app().weight_df
        # restrict to last 3 weeks
        todays_date = dt.datetime.now()
        filtered_df = df[pd.to_datetime(df["datetime"]) > todays_date - dt.timedelta(weeks = 3)]
        if len(filtered_df) == 0:
            return 0

        filtered_df['datetime_numeric'] = pd.to_datetime(filtered_df['datetime']).astype('int64') / 1e9  # Convert to seconds
        slope, _, _, _, _ = linregress(filtered_df['datetime_numeric'], filtered_df['rolling_avg'])
        return round(slope, 2)


class TableGraphSelector(BoxLayout):
    def __init__(self, content_widget, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "horizontal"
        self.padding = 0
        self.spacing = 0

        # Reference to the content widget where table/graph will be shown
        self.content_widget = content_widget

        self.table_button = Button(text="Table", size_hint = (0.5, 1), state="down")
        self.graph_button = Button(text="Graph", size_hint = (0.5, 1))
        
        self.table_button.bind(on_press=self.show_table)
        self.graph_button.bind(on_press=self.show_graph)

        self.add_widget(self.table_button)
        self.add_widget(self.graph_button)


    def show_table(self, instance):
        # Ensure only the Table button is pressed
        self.table_button.state = "down"
        self.graph_button.state = "normal"
        # Update content to show the table
        self.content_widget.show_table()

    def show_graph(self, instance):
        # Ensure only the Graph button is pressed
        self.table_button.state = "normal"
        self.graph_button.state = "down"
        # Update content to show the graph
        self.content_widget.show_graph()        

class TableGraphContent(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.show_table() 

    def show_table(self):
        self.clear_widgets()

        scrollview = ScrollView()
        table_layout = GridLayout(cols=3, size_hint_y=None)
        table_layout.bind(minimum_height=table_layout.setter('height'))

        for i in range(4):
            for j in range(3):
                table_layout.add_widget(Label(text=f'Row {i+1}, Col {j+1}'))

        scrollview.add_widget(table_layout)
        self.add_widget(scrollview)

    def show_graph(self):
        self.clear_widgets()

        graph = Graph(xlabel='X', ylabel='Y', x_ticks_minor=5, x_ticks_major=25,
                      y_ticks_minor=1, y_ticks_major=10, y_grid_label=True,
                      x_grid_label=True, padding=5, x_grid=True, y_grid=True, 
                      xmin=-0, xmax=100, ymin=-1, ymax=1)

        plot = MeshLinePlot(color=[1, 0, 0, 1])
        plot.points = [(x, 0.5 * x % 1) for x in range(0, 101)]
        graph.add_plot(plot)

        self.add_widget(graph)


class WeightScribeApp(App):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weight_df = None  

    def load_dataframe(self):
        self.weight_df = pd.read_csv("src/data/weight_data.csv")

    def build(self):
        self.title = 'WeightScribe'
        self.load_dataframe()


        # start with splitting the title bar and app content
        title_body_layout = BoxLayout(orientation='vertical', padding=0, spacing=0)
        title_label = Label(text="WeightScribe", size_hint = (1, 0.1))
        title_body_layout.add_widget(title_label)

        # add the body 
        body_layout = BoxLayout(orientation = "vertical", padding=0, spacing=0, size_hint = (1, 0.8))
        title_body_layout.add_widget(body_layout)

        # body layout split into callout, table/graph selector, table/graph, upload/download

        # callout
        callout_label = CalloutLabel(size_hint = (1, 0.1))
        body_layout.add_widget(callout_label)

        # table graph
        table_graph_content = TableGraphContent(size_hint = (1, 0.65))

        table_graph_selector = TableGraphSelector(size_hint = (1, 0.05), content_widget=table_graph_content)
        body_layout.add_widget(table_graph_selector)
        body_layout.add_widget(table_graph_content)

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