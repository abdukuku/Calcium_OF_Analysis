import sys
import pandas as pd
import numpy as np
from PyQt5.QtWidgets import (
    QApplication,
    QMainWindow,
    QFileDialog,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QLabel,
    QGridLayout,
    QMessageBox,
    QTableWidget,
    QTableWidgetItem,
    QHBoxLayout,
    QSlider,
)
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import matplotlib.pyplot as plt


class CalciumImagingGUI(QMainWindow):
    """
    A graphical user interface for calcium imaging analysis.

    This class provides a GUI for loading calcium imaging data.
    It allows users to load data, apply filters, visualize data, and display results.
    """

    def __init__(self):
        super().__init__()
        self.df = None
        self.filtered_df = None
        self.initUI()

    # Rest of the code...
class CalciumImagingGUI(QMainWindow):
    
    def __init__(self):
        super().__init__()
        self.df = None
        self.filtered_df = None
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Calcium Imaging Analysis")
        self.setGeometry(100, 100, 1200, 800)

        # File loading section
        self.loadButton = QPushButton("Load Data", self)
        self.loadButton.clicked.connect(self.load_data)

        # Sliders for filtering
        self.pf_pvalue_slider, self.pf_pvalue_label = self.create_slider("PF p-value", 0, 100, 5)
        self.speed_pvalue_slider, self.speed_pvalue_label = self.create_slider("Speed p-value", 0, 100, 5)

        # Data visualization section
        self.neuronLocCanvas = PlotCanvas(self, interact=True)
        self.activateModCanvas = PlotCanvas(self)
        self.speedModCanvas = PlotCanvas(self)
        self.placeFieldCanvases = [PlotCanvas(self) for _ in range(6)]

        # Table widget
        self.tableWidget = QTableWidget()
        self.tableWidget.setFixedSize(300, 300)

        # Layout
        layout = QGridLayout()
        layout.addWidget(self.loadButton, 0, 0)
        layout.addWidget(QLabel("PF p-value"), 0, 1)
        layout.addWidget(self.pf_pvalue_slider, 0, 2)
        layout.addWidget(self.pf_pvalue_label, 0, 3)
        layout.addWidget(QLabel("Speed p-value"), 0, 4)
        layout.addWidget(self.speed_pvalue_slider, 0, 5)
        layout.addWidget(self.speed_pvalue_label, 0, 6)

        # First row: 1:1 and 1:3
        layout.addWidget(self.neuronLocCanvas, 1, 0)
        layout.addWidget(self.activateModCanvas, 1, 1, 1, 5)

        # Second row: 1:1 and 1:3
        layout.addWidget(self.tableWidget, 2, 0)
        layout.addWidget(self.speedModCanvas, 2, 1, 1, 5)

        # Third row: 1:1, 1:1, 1:1, 1:1
        for i, canvas in enumerate(self.placeFieldCanvases):
            layout.addWidget(canvas, 3, i)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def create_slider(self, label, min_val, max_val, init_val):
        slider = QSlider(Qt.Horizontal)
        slider.setRange(min_val, max_val)
        slider.setValue(init_val)
        value_label = QLabel(f"{init_val}")
        slider.valueChanged.connect(lambda value, lbl=value_label: self.update_slider_label(lbl, value))
        slider.valueChanged.connect(self.update_filters)
        return slider, value_label

    def update_slider_label(self, label, value):
        label.setText(f"{value / 100.0:.2f}")

    def load_data(self):
        options = QFileDialog.Options()
        file_name = QFileDialog.getOpenFileName(self, "Open Data File", "", "Pickle Files (*.pkl);;All Files (*)", options=options)[0]
        if file_name:
            self.df = pd.read_pickle(file_name)
            self.generate_neuron_coordinates()
            self.update_filters()

    def generate_neuron_coordinates(self):
        np.random.seed(42)  # For reproducibility
        self.df["NeuronX"] = self.df.groupby("CellID")["CellID"].transform(lambda x: np.random.randint(0, 43))
        self.df["NeuronY"] = self.df.groupby("CellID")["CellID"].transform(lambda x: np.random.randint(0, 43))

    def update_filters(self):
        pf_pvalue_threshold = self.pf_pvalue_slider.value() / 100.0
        speed_pvalue_threshold = self.speed_pvalue_slider.value() / 100.0

        self.pf_pvalue_label.setText(f"{pf_pvalue_threshold:.2f}")
        self.speed_pvalue_label.setText(f"{speed_pvalue_threshold:.2f}")

        self.filtered_df = self.df[(self.df["PF_pvalue"] <= pf_pvalue_threshold) & (self.df["Speed_pvalue"] <= speed_pvalue_threshold)]
        self.display_data()

    def display_data(self):
        self.plot_neuron_locations()
        self.plot_additional_data()

    def plot_neuron_locations(self):
        self.neuronLocCanvas.plot_scatter(
            "NeuronX", "NeuronY", "Neuron Locations", "Neuron Location (x)", "Neuron Location (y)", self.df, self.filtered_df
        )
    def plot_additional_data(self):
        self.placeFieldCanvases[4].plot_hist(
            self.filtered_df["eventAmpMedian"], "Event Amplitude Distribution", "Event Amplitude", "Frequency"
        )
        self.placeFieldCanvases[5].plot_hist(self.filtered_df["PFsize"], "Place Field Size Distribution", "Place Field Size", "Frequency")
    
    def populate_table(self, neuron_index):
        measure_of_interest = [
            "Frate",
            "Frate_moving",
            "Frate_rest",
            "PF_pvalue",
            "PF_SI",
            "PFsize",
            "Sparsity",
            "Selectivity",
            "Speed_SI",
            "Speed_pvalue",
        ]
        neuron_data = self.filtered_df[self.filtered_df["CellID"] == neuron_index]
        neuron_data.set_index("Day", inplace=True)
        neuron_data = neuron_data[measure_of_interest].T.round(2)
        nrows, ncols = neuron_data.shape
        self.tableWidget.setRowCount(nrows)
        self.tableWidget.setColumnCount(ncols + 1)  # +1 for the index column

        # Set the column headers, including one for the index
        headers = ["Measure"] + list(neuron_data.columns)
        self.tableWidget.setHorizontalHeaderLabels(headers)

        # Populate the table with the DataFrame's content, including the index
        for row in range(nrows):
            # Set the index item
            index_item = QTableWidgetItem(str(neuron_data.index[row]))
            self.tableWidget.setItem(row, 0, index_item)

            for col in range(ncols):
                item = QTableWidgetItem(str(neuron_data.iloc[row, col]))
                self.tableWidget.setItem(row, col + 1, item)  # +1 because the first column is now the index

    def plot_snr_eventRate(self, neuron_index):
        self.placeFieldCanvases[3].plot_scatter(
            "snr", "eventRate", "Quality Plot", "snr", "eventRate [Hz]", self.df, self.filtered_df
        )
        neuron_data = self.filtered_df[self.filtered_df["CellID"] == neuron_index]
        selected_dot = pg.ScatterPlotItem(neuron_data["snr"], neuron_data["eventRate"], size=15, pen=pg.mkPen("r", width=2), brush=pg.mkBrush(255, 0, 0, 120))
        self.placeFieldCanvases[3].plot.addItem(selected_dot)
        
    def plot_speed_modulation(self, neuron_index):
        neuron_data = self.filtered_df[self.filtered_df["CellID"] == neuron_index]
        self.speedModCanvas.plot_speed(neuron_data, "Speed Modulation", "Speed (cm/s)", "AUC")

    def plot_activate_modulation(self, neuron_index):
        neuron_data = self.filtered_df[self.filtered_df["CellID"] == neuron_index]
        self.activateModCanvas.plot_trace(neuron_data, "Fluorescent Trace", "Time(s)", "AUC")

    def plot_place_fields(self, neuron_index):
        days = ["HD1", "HD2", "HD3"]
        for i, day in enumerate(days):
            neuron_data = self.filtered_df[(self.filtered_df["CellID"] == neuron_index) & (self.filtered_df["Day"] == day)]
            self.placeFieldCanvases[i].plot_imshow(neuron_data, f"Place Field {day}")


class PlotCanvas(pg.GraphicsLayoutWidget):
    def __init__(self, parent=None, interact=False):
        super().__init__(parent)
        self.parent = parent
        self.df = None
        self.plot = self.addPlot()
        self.plot.showGrid(x=True, y=True)
        self.selected_dot = None
        if interact:
            self.plot.scene().sigMouseClicked.connect(self.on_click)

    def plot_scatter(self, x, y, title, xlabel, ylabel, df=None, filtered_df=None):
        self.plot.clear()
        self.df = df

        scatter_data = []
        colors = {"HD1": "m", "HD2": "y", "HD3": "c"}

        for day, color in colors.items():
            day_data = filtered_df[filtered_df["Day"] == day]
            scatter_data.append(
                pg.ScatterPlotItem(x=day_data[x], y=day_data[y], size=10, pen=pg.mkPen(color=color), brush=pg.mkBrush(color))
            )

        grey_data = df[~df.index.isin(filtered_df.index)]
        scatter_data.append(
            pg.ScatterPlotItem(x=grey_data[x], y=grey_data[y], size=10, pen=pg.mkPen(None), brush=pg.mkBrush(128, 128, 128, 80))
        )

        for scatter in scatter_data:
            self.plot.addItem(scatter)

        if self.selected_dot is not None:
            self.plot.addItem(self.selected_dot)

        self.plot.setTitle(title)
        self.plot.setLabel("left", ylabel)
        self.plot.setLabel("bottom", xlabel)

    def plot_trace(self, data, title, xlabel, ylabel):
        self.plot.clear()
        colors = {"HD1": "m", "HD2": "y", "HD3": "c"}
        for _, neuron in data.iterrows():
            filtered_spikes = neuron["Spikes"][neuron["Spikes"] != 0]
            day = neuron["Day"]
            color = colors.get(day, "r")  # default to red if day not in dict
            self.plot.plot(neuron["RawTrace"].index.values, neuron["RawTrace"].values, pen=color)
            self.plot.plot(filtered_spikes.index.values, neuron["RawTrace"][filtered_spikes.index].values, pen=None, symbol="o", symbolBrush="w")
        self.plot.setTitle(title)
        self.plot.setLabel("left", ylabel)
        self.plot.setLabel("bottom", xlabel)

    def plot_speed(self, data, title, xlabel, ylabel):
        self.plot.clear()
        colors = {"HD1": "m", "HD2": "y", "HD3": "c"}
        for _, neuron in data.iterrows():
            speed = list(neuron["Speed_tcurve"].keys())
            AUC = list(neuron["Speed_tcurve"].values())
            day = neuron["Day"]
            color = colors.get(day, "w")  # default to white if day not in dict
            self.plot.plot(speed, AUC, pen=color)
        self.plot.setTitle(title)
        self.plot.setLabel("left", ylabel)
        self.plot.setLabel("bottom", xlabel)

    def plot_hist(self, data, title, xlabel, ylabel):
        self.plot.clear()
        y, x = np.histogram(data, bins=30)
        self.plot.plot(x, y, stepMode=True, fillLevel=0, brush=pg.mkBrush(0, 255, 0, 120))
        self.plot.setTitle(title)
        self.plot.setLabel("left", ylabel)
        self.plot.setLabel("bottom", xlabel)

    def generate_contours(self, array, levels=0):
        cs = plt.contour(array, levels=levels)
        return cs.allsegs

    def plot_imshow(self, data, title):
        self.plot.clear()
        if len(data) > 0:
            img = pg.ImageItem(image=data.iloc[0]["Turning"])
            contours = self.generate_contours(data.iloc[0]["PFmask"])
            self.plot.addItem(img)
            for level_segs in contours:
                for seg in level_segs:
                    seg = np.array(seg)
                    curve = pg.PlotCurveItem(seg[:, 1], seg[:, 0], pen=pg.mkPen(color='r', width=1))
                    self.plot.addItem(curve)
            self.plot.plot()
            self.plot.addColorBar(img, colorMap="viridis", values=(0, .7))
            self.plot.setTitle(title)

    def on_click(self, event):
        if self.df is not None:
            pos = event.scenePos()
            mouse_point = self.plot.vb.mapSceneToView(pos)
            distances = np.sqrt((self.df["NeuronX"] - mouse_point.x()) ** 2 + (self.df["NeuronY"] - mouse_point.y()) ** 2)
            closest_index = distances.idxmin()
            closest_neuron = self.df.loc[closest_index]

            # Display the place field and information of the clicked neuron
            self.parent.plot_speed_modulation(closest_neuron["CellID"])
            self.parent.plot_snr_eventRate(closest_neuron["CellID"])
            self.parent.plot_activate_modulation(closest_neuron["CellID"])
            self.parent.plot_place_fields(closest_neuron["CellID"])
            self.parent.populate_table(closest_neuron["CellID"])

            # Highlight the selected dot
            if self.selected_dot is not None:
                self.plot.removeItem(self.selected_dot)

            self.selected_dot = pg.ScatterPlotItem(
                x=[closest_neuron["NeuronX"]], y=[closest_neuron["NeuronY"]], size=15, pen=pg.mkPen("r", width=2), brush=pg.mkBrush(255, 0, 0, 120)
            )
            self.plot.addItem(self.selected_dot)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = CalciumImagingGUI()
    ex.show()
    sys.exit(app.exec_())
