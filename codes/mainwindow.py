import sys
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import computational_metrics_new  # Added import for computational metrics visualization
import types  # Added for dynamic method assignment

from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QLabel, QDoubleSpinBox, QSpinBox,
    QVBoxLayout, QHBoxLayout, QPushButton, QTabWidget, QFormLayout, QGroupBox,
    QTextEdit, QCheckBox, QScrollArea, QFileDialog, QMessageBox, QDockWidget,
    QMenuBar, QMenu, QAction, QSplitter, QToolBar, QStatusBar, QLineEdit, QComboBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView, QSizePolicy, QActionGroup,
    QStackedWidget, QFrame, QListWidget, QListWidgetItem, QApplication, QGraphicsDropShadowEffect,
    QRadioButton, QButtonGroup, QGridLayout, QDialogButtonBox, QDialog, QGraphicsOpacityEffect,
    QToolButton, QStyle, QStyledItemDelegate, QProgressBar, QInputDialog, QColorDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSize, QPoint, QPropertyAnimation, QEasingCurve, QRect, QTimer, QDateTime
from PyQt5.QtGui import QIcon, QPalette, QColor, QFont, QPixmap, QCursor, QPainter, QBrush, QLinearGradient, QMovie
from PyQt5.QtGui import QGuiApplication

# Matplotlib backends
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar

# Local imports from "modules" subfolder
from modules.FRF import frf
from modules.plotwindow import PlotWindow

# Local imports from "workers" subfolder
from workers.FRFWorker import FRFWorker
from workers.GAWorker import GAWorker
from workers.PSOWorker import PSOWorker, TopologyType
from workers.DEWorker import DEWorker
from workers.SAWorker import SAWorker
from workers.CMAESWorker import CMAESWorker
# RL functionality is provided through RLOptimizationMixin

from gui.menu_mixin import MenuMixin
from gui.beam_mixin import ContinuousBeamMixin
from gui.microchip_mixin import MicrochipPageMixin
from gui.main_window.theme_mixin import ThemeMixin
from gui.main_window.ga_mixin import GAOptimizationMixin
from gui.main_window.frf_mixin import FRFMixin
from gui.main_window.pso_mixin import PSOMixin
# Import the DE mixin separately - we'll handle its integration differently
from gui.main_window.de_mixin import DEOptimizationMixin
from gui.main_window.input_mixin import InputTabsMixin
from gui.main_window.extra_opt_mixin import ExtraOptimizationMixin
from gui.main_window.sidebar_mixin import SidebarMixin
from gui.main_window.stochastic_mixin import StochasticDesignMixin
from gui.main_window.sobol_mixin import SobolAnalysisMixin
from gui.main_window.omega_sensitivity_mixin import OmegaSensitivityMixin

from app_info import APP_NAME, __version__

# Additional libraries used
import random
from deap import base, creator, tools

# Seaborn style and LaTeX rendering
sns.set(style="whitegrid")
plt.rc('text', usetex=True)

# Import Continuous Beam functionality (new API)
try:
    from Continues_beam import create_beam_optimization_interface
    BEAM_IMPORTS_SUCCESSFUL = True
except ImportError:
    BEAM_IMPORTS_SUCCESSFUL = False
    print("Warning: Continuous Beam module imports failed. This functionality will be disabled.")


        

class MainWindow(QMainWindow, MenuMixin, ContinuousBeamMixin, MicrochipPageMixin,
                 ThemeMixin, FRFMixin, PSOMixin, GAOptimizationMixin,
                 InputTabsMixin, ExtraOptimizationMixin,
                 SidebarMixin, StochasticDesignMixin, SobolAnalysisMixin, OmegaSensitivityMixin):
    # Track all open MainWindow instances for Playground tiling/management
    playground_windows = []
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} - {__version__}")
        self.resize(1600, 900)
        self.setMinimumSize(1200, 800)  # Set minimum window size
        
        # Disable LaTeX rendering in matplotlib to prevent Unicode errors with Greek characters
        import matplotlib as mpl
        mpl.rcParams['text.usetex'] = False
        
        # Initialize theme
        self.current_theme = 'Dark'  # Changed to default dark theme
        
        # Initialize variables for comparative visualization
        self.available_plots_list = None
        self.legend_table = None
        self.legend_map = {}
        self.frf_plots = {}
        self.comp_fig = None
        self.comp_canvas = None
        
        # Create central widget with main layout
        central_widget = QWidget()
        central_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.main_layout = QHBoxLayout(central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.setCentralWidget(central_widget)
        
        # Create sidebar
        self.create_sidebar(BEAM_IMPORTS_SUCCESSFUL)
        
        # Create scroll area for content
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        # Create stacked widget for main content
        self.content_stack = QStackedWidget()
        self.content_stack.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        scroll_area.setWidget(self.content_stack)
        self.main_layout.addWidget(scroll_area, 1)
        
        # Create the various content pages
        self.create_stochastic_design_page()
        
        # Integrate DE functionality before showing the window
        self.integrate_de_functionality()
        
        self.create_microchip_controller_page()
        self.create_continuous_beam_page()
        
        # Set default active page
        self.content_stack.setCurrentIndex(0)
        
        # Apply theme
        self.apply_dark_theme()
        
        # Create status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # Create menubar and toolbar
        self.create_menubar()
        self.create_toolbar()
        
        self.current_ga_best_params = None
        self.current_ga_best_fitness = None
        self.current_ga_full_results = None
        self.current_ga_settings = None

        # Initialize other optimization results holders if they follow a similar pattern
        self.current_pso_best_params = None
        self.current_rl_best_params = None
        self.current_rl_best_reward = None
        
        # Initialize zones for FRF comparative plotting
        self.zones = []

        # Playground: register instance and default name
        try:
            if self not in MainWindow.playground_windows:
                MainWindow.playground_windows.append(self)
        except Exception:
            pass
        self.instance_name = ""
        self._update_window_title()

    def integrate_de_functionality(self):
        """Manually integrate DE functionality from DEOptimizationMixin"""
        try:
            print("Starting DE functionality integration...")
            # Import necessary methods from DEOptimizationMixin
            # Create a temporary instance to access methods
            temp_de_mixin = DEOptimizationMixin()
            
            # Import methods by copying their bound versions to this instance
            method_list = [
                'run_de', 'initialize_de_parameter_table', 'get_parameter_data',
                'toggle_de_fixed', 'handle_de_progress', 'update_de_visualization',
                'handle_de_finished', 'create_de_final_visualization',
                'save_de_visualization', 'handle_de_error', 'handle_de_update',
                'tune_de_hyperparameters'
            ]
            
            for method_name in method_list:
                if hasattr(DEOptimizationMixin, method_name):
                    method = getattr(DEOptimizationMixin, method_name)
                    setattr(self, method_name, types.MethodType(method, self))
            
            print("Methods imported successfully")
            
            # Create the DE tab properly
            if hasattr(self, 'de_tab'):
                print(f"DE tab exists: {self.de_tab}")
                # If the tab exists, remove it first
                if hasattr(self, 'optimization_tabs'):
                    for i in range(self.optimization_tabs.count()):
                        tab_text = self.optimization_tabs.tabText(i)
                        if tab_text == "DE Optimization" or tab_text == "DE Optimization (Placeholder)":
                            print(f"Removing existing DE tab at index {i}: {tab_text}")
                            self.optimization_tabs.removeTab(i)
                            break
            else:
                print("DE tab doesn't exist yet")
            
            # Now create the DE tab using the imported method
            print("Creating DE tab...")
            self.create_de_tab = types.MethodType(DEOptimizationMixin.create_de_tab, self)
            
            try:
                # Create the DE tab with proper error handling
                new_de_tab = self.create_de_tab()
                print(f"New DE tab created: {new_de_tab}")
                if new_de_tab is not None:
                    self.de_tab = new_de_tab
                else:
                    print("Warning: create_de_tab returned None")
                    # Create a simple tab as fallback
                    self.de_tab = QWidget()
                    fallback_layout = QVBoxLayout(self.de_tab)
                    fallback_label = QLabel("DE Optimization Tab (Fallback)")
                    fallback_layout.addWidget(fallback_label)
            except Exception as e:
                print(f"Error creating DE tab: {str(e)}")
                # Create a simple tab as fallback
                self.de_tab = QWidget()
                fallback_layout = QVBoxLayout(self.de_tab)
                fallback_label = QLabel(f"Error creating DE tab: {str(e)}")
                fallback_layout.addWidget(fallback_label)
            
            print(f"Final DE tab: {self.de_tab}")
            
            # Ensure it's added to the optimization_tabs
            if hasattr(self, 'optimization_tabs'):
                print(f"Adding DE tab to optimization_tabs with {self.optimization_tabs.count()} existing tabs")
                if self.de_tab:
                    tab_index = self.optimization_tabs.addTab(self.de_tab, "DE Optimization")
                    print(f"DE tab added at index {tab_index}")
                else:
                    print("Cannot add DE tab because it is None")
                
                # Verify tab was added
                de_found = False
                for i in range(self.optimization_tabs.count()):
                    tab_text = self.optimization_tabs.tabText(i)
                    print(f"Tab at index {i}: {tab_text}")
                    if tab_text == "DE Optimization":
                        de_found = True
                        # Ensure the tab is visible
                        self.de_tab.setVisible(True)
                
                print(f"DE tab found in tabs: {de_found}")
                
                # Don't automatically switch to the DE tab - let the user choose
                # The default should be the Input tab
                
            else:
                print("optimization_tabs not found")
            
            print("DE functionality integration successful")
            
        except Exception as e:
            print(f"Failed to integrate DE functionality: {str(e)}")
            # Don't raise the exception, just report it
            import traceback
            traceback.print_exc()

    def set_default_values(self):
        """Reset all inputs to their default values"""
        self.status_bar.showMessage("Resetting to default values...")
        
        try:
            # Reset PSO parameters
            self.pso_swarm_size_box.setValue(40)
            self.pso_num_iterations_box.setValue(100)
            self.pso_inertia_box.setValue(0.729)
            self.pso_cognitive_box.setValue(1.49445)
            self.pso_social_box.setValue(1.49445)
            self.pso_tol_box.setValue(1e-6)
            self.pso_alpha_box.setValue(0.01)
            self.pso_benchmark_runs_box.setValue(1)
            
            # Reset advanced PSO parameters
            self.pso_adaptive_params_checkbox.setChecked(True)
            self.pso_topology_combo.setCurrentText("Global")
            self.pso_w_damping_box.setValue(1.0)
            self.pso_mutation_rate_box.setValue(0.1)
            self.pso_max_velocity_factor_box.setValue(0.1)
            self.pso_stagnation_limit_box.setValue(10)
            self.pso_boundary_handling_combo.setCurrentText("absorbing")
            self.pso_diversity_threshold_box.setValue(0.01)
            self.pso_early_stopping_checkbox.setChecked(True)
            self.pso_early_stopping_iters_box.setValue(15)
            self.pso_early_stopping_tol_box.setValue(1e-5)
            self.pso_quasi_random_init_checkbox.setChecked(True)
            
            # Reset DVA parameters in PSO table
            for row in range(self.pso_param_table.rowCount()):
                param_name = self.pso_param_table.item(row, 0).text()
                # Uncheck fixed checkbox
                fixed_checkbox = self.pso_param_table.cellWidget(row, 1)
                fixed_checkbox.setChecked(False)
                
                # Reset fixed value
                fixed_value_spin = self.pso_param_table.cellWidget(row, 2)
                fixed_value_spin.setValue(0.0)
                
                # Reset bounds based on parameter type
                lower_bound_spin = self.pso_param_table.cellWidget(row, 3)
                upper_bound_spin = self.pso_param_table.cellWidget(row, 4)
                
                if param_name.startswith(("beta_", "lambda_", "nu_")):
                    lower_bound_spin.setValue(0.0001)
                    upper_bound_spin.setValue(10.0)
                elif param_name.startswith("mu_"):
                    lower_bound_spin.setValue(0.0)
                    upper_bound_spin.setValue(1.0)
            
            # Reset FRF parameters
            self.omega_start_box.setValue(0.1)
            self.omega_end_box.setValue(10.0)
            self.omega_points_box.setValue(1000)
            
            # Reset comparative visualization options
            self.plot_title_edit.clear()
            self.title_font_size.setValue(14)
            self.fig_width_spin.setValue(10)
            self.fig_height_spin.setValue(6)
            self.x_norm_check.setChecked(False)
            self.y_norm_check.setChecked(False)
            self.x_norm_value.setValue(1.0)
            self.y_norm_value.setValue(1.0)
            self.show_grid_check.setChecked(True)
            self.legend_position_combo.setCurrentText("best")
            
            # Clear plots and results
            self.available_plots_list.clear()
            self.legend_table.setRowCount(0)
            self.legend_map.clear()
            self.frf_plots.clear()
            
            # Reset optimization results
            self.current_ga_best_params = None
            self.current_ga_best_fitness = None
            self.current_ga_full_results = None
            self.current_ga_settings = None
            self.current_pso_best_params = None
            self.current_rl_best_params = None
            self.current_rl_best_reward = None
            
            self.status_bar.showMessage("All values reset to defaults", 3000)
            
        except Exception as e:
            QMessageBox.warning(self, "Reset Error", 
                              f"Error resetting some values: {str(e)}\nSome values may not have been reset.")
            self.status_bar.showMessage("Error during reset", 3000)



    # ---------------------------
    # Playground support methods
    # ---------------------------
    def _update_window_title(self):
        try:
            base = f"{APP_NAME} - {__version__}"
            if hasattr(self, 'instance_name') and self.instance_name:
                self.setWindowTitle(f"{base} — {self.instance_name}")
            else:
                self.setWindowTitle(base)
        except Exception:
            pass

    def set_instance_name(self, name):
        try:
            self.instance_name = str(name).strip()
            self._update_window_title()
        except Exception:
            pass

    def rename_instance_window(self):
        try:
            text, ok = QInputDialog.getText(self, "Rename Window", "Instance name:", text=self.instance_name or "")
            if ok:
                self.set_instance_name(text)
        except Exception:
            pass

    def open_playground_clone(self):
        try:
            # Prompt for new instance name
            name, ok = QInputDialog.getText(self, "New Playground Window", "Name this instance:")
            if not ok:
                return

            # Collect current parameters
            params = {}
            try:
                if hasattr(self, '_collect_parameters_to_dict'):
                    params = self._collect_parameters_to_dict()
            except Exception:
                params = {}

            # Create and configure clone window
            clone = MainWindow()

            # Apply theme/state
            try:
                clone.switch_theme(self.current_theme)
            except Exception:
                pass

            # Apply parameters
            try:
                if params and hasattr(clone, '_apply_parameters_from_dict'):
                    clone._apply_parameters_from_dict(params)
            except Exception:
                pass

            # Name and show
            clone.set_instance_name(name)
            clone.show()
            clone.raise_()
            clone.activateWindow()

            # Stagger position slightly
            try:
                g = self.geometry()
                clone.setGeometry(g.adjusted(40, 40, 40, 40))
            except Exception:
                pass
        except Exception as e:
            try:
                QMessageBox.warning(self, "Playground Error", f"Failed to open clone: {str(e)}")
            except Exception:
                pass

    def tile_playground_windows(self):
        try:
            windows = [w for w in getattr(MainWindow, 'playground_windows', []) if w is not None and w.isVisible()]
            if not windows:
                return

            # Determine grid layout
            n = len(windows)
            import math
            cols = int(math.ceil(math.sqrt(n)))
            rows = int(math.ceil(n / cols))

            # Available geometry
            screen = QGuiApplication.primaryScreen()
            if not screen:
                return
            avail = screen.availableGeometry()
            cell_w = max(400, int(avail.width() / cols))
            cell_h = max(300, int(avail.height() / rows))

            for idx, w in enumerate(windows):
                r = idx // cols
                c = idx % cols
                x = avail.x() + c * cell_w
                y = avail.y() + r * cell_h
                try:
                    w.setGeometry(x, y, cell_w, cell_h)
                except Exception:
                    pass
        except Exception:
            pass

    def _playground_close_cleanup(self):
        try:
            if hasattr(MainWindow, 'playground_windows') and self in MainWindow.playground_windows:
                MainWindow.playground_windows.remove(self)
        except Exception:
            pass

