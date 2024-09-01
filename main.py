from kivy.config import Config
if __name__ == '__main__':
    width = 360
    height = int((2000/1080) * width)
    Config.set('graphics', 'width', f'{width}')
    Config.set('graphics', 'height', f'{height}')

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.anchorlayout import AnchorLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.label import Label
from plyer import filechooser
from kivy.clock import mainthread
from kivy.uix.scrollview import ScrollView
from kivy.core.window import Window
import pandas as pd
import datetime as dt
from scipy.stats import linregress
from kivy_garden.graph import Graph, MeshLinePlot, LinePlot, ScatterPlot
import src.colours as colours


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

class BinaryButton(BoxLayout):
    def __init__(self, table_widget, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "horizontal"
        self.table_widget = table_widget

        self.weight_button = Button(text="Weight", size_hint = (0.5, 1), state="down")
        self.MA_button = Button(text="MA Weight", size_hint = (0.5, 1))

        self.weight_button.bind(on_press=self.change_weight)
        self.MA_button.bind(on_press=self.change_MA_weight)

        self.add_widget(self.weight_button)
        self.add_widget(self.MA_button)

    def change_weight(self, instance):
        self.weight_button.state = "down"
        self.MA_button.state = "normal"
        self.table_widget.rolling_view = False
        self.table_widget.populate_table()

    def change_MA_weight(self, instance):
        self.weight_button.state = "normal"
        self.MA_button.state = "down"
        self.table_widget.rolling_view = True
        self.table_widget.populate_table()
        
class Table(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.rolling_view = False
        self.df = App.get_running_app().weight_df

        table_binary_layout = BoxLayout(orientation="horizontal", size_hint = (1, 0.1))
        weight_selection_label = Label(text="Weight", size_hint_x = None)
        weight_binary_button = BinaryButton(size_hint_x = None, table_widget=self)
        table_binary_spacer = Label(size_hint_x = 1)
        table_binary_layout.add_widget(weight_selection_label)
        table_binary_layout.add_widget(table_binary_spacer)
        table_binary_layout.add_widget(weight_binary_button)


        self.add_widget(table_binary_layout)

        table_header_layout = BoxLayout(orientation="horizontal", size_hint = (1, 0.15))
        table_header_layout.add_widget(Label(text="Date"))
        table_header_layout.add_widget(Label(text="Weight"))
        self.add_widget(table_header_layout)

        scrollview = ScrollView()
        self.table_layout = GridLayout(cols=2, size_hint_y=None)
        self.table_layout.bind(minimum_height=self.table_layout.setter('height'))

        self.populate_table()

        scrollview.add_widget(self.table_layout)
        self.add_widget(scrollview)

    def populate_table(self):
        self.table_layout.clear_widgets()
        weight_column_used = "weight" if self.rolling_view == False else "rolling_avg"

        for _, row in self.df.iterrows():
            self.table_layout.add_widget(Label(text=str(row['date'])))
            self.table_layout.add_widget(Label(text=str(row[weight_column_used])))
        window_height = Window.height
        row_height = window_height * 0.05
        self.table_layout.height = len(self.df) * row_height

class WeightGraphWidget(Graph):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        df = App.get_running_app().weight_df

        todays_date = dt.datetime.now()
        df["days_since_today"] = (todays_date - pd.to_datetime(df["datetime"])).dt.days.astype(int)
        weight_max_min = (int(pd.concat([df["weight"], df["rolling_avg"]]).max()), int(pd.concat([df["weight"], df["rolling_avg"]]).min()))
        max_days = df["days_since_today"].max()

        self.xlabel = "Days from Today"
        self.ylabel = "Weight (kg)"
        self.x_ticks_minor = 4
        self.x_ticks_major = 30
        self.y_ticks_minor = 2
        self.y_ticks_major = 1
        self.y_grid_label = True
        self.x_grid_label = True
        self.padding = 5
        self.x_grid = True
        self.y_grid = True
        self.xmin = 0
        self.xmax = 30
        self.ymin = weight_max_min[1]*0.99
        self.ymax = weight_max_min[0]*1.01

        self.zoom_days = [30, 60, 90, 180, 365, 730]
        self.zoom_days = [days for days in self.zoom_days if days <= max_days] + [int(max_days)]

        scatter_plot = ScatterPlot(color=[1, 0, 0, 1])
        scatter_plot.points = list(zip(df["days_since_today"], df["weight"]))

        line_plot = LinePlot(color=[1, 0, 0, 1])
        line_plot.points = list(zip(df["days_since_today"], df["rolling_avg"]))

        self.add_plot(scatter_plot)
        self.add_plot(line_plot)

    def zoom_in(self, instance):
        if self.xmax <= 30:
            # cant zoom in anymore
            return
        else:
            current_zoom = self.zoom_days.index(self.xmax)
            self.xmax = self.zoom_days[current_zoom-1]

    def zoom_out(self, instance):
        if self.xmax >= max(self.zoom_days):
            # cant zoom out anymore
            return
        else:
            current_zoom = self.zoom_days.index(self.xmax)
            self.xmax = self.zoom_days[current_zoom+1]

class WeightGraph(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        
        # graph and then graph zoom in and out controls
        self.graph_widget = WeightGraphWidget()
        self.add_widget(self.graph_widget)

        graph_controls_layout = BoxLayout(orientation="horizontal", size_hint = (1, 0.1))
        zoom_in_button = Button(text="+")
        zoom_out_button = Button(text="-")

        zoom_in_button.bind(on_press=self.graph_widget.zoom_in)
        zoom_out_button.bind(on_press=self.graph_widget.zoom_out)
        spacing_widget = Label(size_hint_x = 2)

        graph_controls_layout.add_widget(spacing_widget)
        graph_controls_layout.add_widget(zoom_out_button)
        graph_controls_layout.add_widget(zoom_in_button)
        self.add_widget(graph_controls_layout)



class TableGraphContent(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.orientation = "vertical"
        self.show_table() 

    def show_table(self):
        self.clear_widgets()
        table_widget = Table()
        self.add_widget(table_widget)

    def show_graph(self):
        self.clear_widgets()
        graph = WeightGraph()
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
    WeightScribeApp().run()