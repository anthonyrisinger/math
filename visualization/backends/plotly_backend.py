#!/usr/bin/env python3
"""
Plotly Interactive Dashboard Backend
===================================

AGGRESSIVE MATPLOTLIB REPLACEMENT
Modern web-based interactive dashboards with mathematical precision.
Maintains orthographic constraints and control semantics.
"""

from typing import Dict, Any, Optional, List, Tuple, Union
import numpy as np

from .base_backend import VisualizationBackend, CameraConfig

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    import plotly.colors as pc
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    go = px = pc = None


class PlotlyDashboard(VisualizationBackend):
    """
    Modern Plotly-based interactive dashboard.
    COMPLETE MATPLOTLIB REPLACEMENT with extreme prejudice.
    """
    
    def __init__(self):
        super().__init__("plotly")
        self.fig = None
        self.subplot_specs = None
        self.color_sequences = {}
        
    def initialize(self, **kwargs) -> bool:
        """Initialize Plotly dashboard with mathematical layout."""
        if not PLOTLY_AVAILABLE:
            print("WARNING: plotly not installed. Install with: pip install plotly")
            return False
            
        try:
            # Create dashboard with mathematical grid layout (2x3 like original)
            layout = kwargs.get('layout', 'mathematical_grid')
            
            if layout == 'mathematical_grid':
                self._create_mathematical_grid()
            elif layout == 'unified':
                self._create_unified_layout()
            else:
                self._create_default_layout()
            
            # Apply orthographic camera configuration
            self.set_orthographic_projection()
            
            # Set mathematical theme
            self._apply_mathematical_theme()
            
            self._initialized = True
            print(f"✅ Plotly dashboard initialized: {layout}")
            return True
            
        except Exception as e:
            print(f"❌ Plotly initialization failed: {e}")
            return False
    
    def _create_mathematical_grid(self) -> None:
        """Create 2x3 mathematical grid matching original architecture."""
        self.subplot_specs = [
            [{"type": "scene"}, {"type": "xy"}, {"type": "scene"}],
            [{"type": "xy"}, {"type": "xy"}, {"type": "xy"}]
        ]
        
        self.fig = make_subplots(
            rows=2, cols=3,
            specs=self.subplot_specs,
            subplot_titles=[
                "Dimensional Landscape", "Emergence Cascade", "Topology View",
                "Morphic Transformations", "Phase Flow", "Controls & Status"
            ],
            horizontal_spacing=0.08,
            vertical_spacing=0.12
        )
        
    def _create_unified_layout(self) -> None:
        """Create unified single-view layout for focused analysis."""
        self.fig = go.Figure()
        
    def _create_default_layout(self) -> None:
        """Create default flexible layout."""
        self.subplot_specs = [
            [{"type": "scene"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}]
        ]
        
        self.fig = make_subplots(
            rows=2, cols=2,
            specs=self.subplot_specs,
            subplot_titles=["3D View", "Analysis", "Controls", "Status"]
        )
    
    def _apply_mathematical_theme(self) -> None:
        """Apply mathematical styling theme."""
        # Dark mathematical theme
        self.fig.update_layout(
            template="plotly_dark",
            title={
                'text': "Dimensional Emergence Framework - Interactive Dashboard",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 20, 'family': 'Computer Modern, serif'}
            },
            font={'family': 'Computer Modern, monospace', 'size': 12},
            paper_bgcolor='rgba(17,17,17,1)',
            plot_bgcolor='rgba(17,17,17,1)',
            margin=dict(l=20, r=20, t=60, b=20),
        )
        
        # Mathematical color sequences
        self.color_sequences = {
            'gamma': px.colors.sequential.Viridis,
            'morphic': px.colors.sequential.Plasma,
            'topology': px.colors.sequential.Inferno,
            'emergence': px.colors.sequential.Turbo,
            'phase': px.colors.diverging.RdBu,
            'complexity': px.colors.sequential.Cividis
        }
    
    def render_scene(self, scene_data: Dict[str, Any]) -> Any:
        """Render mathematical scene with Plotly precision."""
        if not self._initialized:
            raise RuntimeError("Plotly backend not initialized")
            
        if not self.validate_mathematical_integrity(scene_data):
            raise ValueError("Scene data fails mathematical integrity check")
        
        # Clear existing traces
        self.fig.data = []
        
        # Extract data
        geometry = scene_data.get('geometry', {})
        topology = scene_data.get('topology', {})
        measures = scene_data.get('measures', {})
        parameters = scene_data.get('parameters', {})
        
        # Render components based on layout
        if hasattr(self, 'subplot_specs') and self.subplot_specs:
            self._render_grid_layout(geometry, topology, measures, parameters)
        else:
            self._render_unified_layout(geometry, topology, measures, parameters)
        
        return self.fig
    
    def _render_grid_layout(self, geometry: Dict, topology: Dict, 
                           measures: Dict, parameters: Dict) -> None:
        """Render mathematical grid layout components."""
        
        # (1,1) Dimensional Landscape - 3D Scene
        if 'landscape' in measures:
            self._render_dimensional_landscape(measures['landscape'], row=1, col=1)
        
        # (1,2) Emergence Cascade - 2D Plot  
        if 'cascade' in topology:
            self._render_emergence_cascade(topology['cascade'], row=1, col=2)
        
        # (1,3) Topology View - 3D Scene
        if 'topology_view' in geometry:
            self._render_topology_view(geometry['topology_view'], row=1, col=3)
        
        # (2,1) Morphic Transformations - 2D Plot
        if 'morphic' in geometry:
            self._render_morphic_transformations(geometry['morphic'], row=2, col=1)
        
        # (2,2) Phase Flow - 2D Plot
        if 'phase_flow' in topology:
            self._render_phase_flow(topology['phase_flow'], row=2, col=2)
        
        # (2,3) Controls & Status - Text/Indicators
        if 'controls' in parameters:
            self._render_controls_status(parameters['controls'], row=2, col=3)
    
    def _render_dimensional_landscape(self, landscape_data: Dict, row: int, col: int) -> None:
        """Render dimensional landscape with orthographic 3D."""
        d_range = landscape_data.get('d_range', np.linspace(0.1, 12, 100))
        volumes = landscape_data.get('volumes', [])
        surfaces = landscape_data.get('surfaces', [])
        complexity = landscape_data.get('complexity', [])
        current_d = landscape_data.get('current_dimension', 4.0)
        
        # Create 3D surface plot
        X, Y = np.meshgrid(d_range[:50], d_range[:50])  # Grid for surface
        Z = np.outer(volumes[:50], surfaces[:50]) if len(volumes) >= 50 and len(surfaces) >= 50 else np.zeros_like(X)
        
        # 3D Surface
        surface = go.Surface(
            x=X, y=Y, z=Z,
            colorscale=self.color_sequences['gamma'],
            opacity=0.8,
            name="Dimensional Surface"
        )
        self.fig.add_trace(surface, row=row, col=col)
        
        # Current dimension marker
        if len(volumes) > 0 and len(surfaces) > 0:
            current_idx = min(int(current_d * 10), len(volumes) - 1)
            current_vol = volumes[current_idx] if current_idx < len(volumes) else 1.0
            current_surf = surfaces[current_idx] if current_idx < len(surfaces) else 1.0
            
            marker = go.Scatter3d(
                x=[current_d], y=[current_d], z=[current_vol * current_surf],
                mode='markers',
                marker=dict(size=10, color='red', symbol='diamond'),
                name=f"Current d={current_d:.1f}"
            )
            self.fig.add_trace(marker, row=row, col=col)
        
        # Apply orthographic camera
        scene_name = f"scene{'' if row == 1 and col == 1 else str((row-1)*3 + col)}"
        self._apply_orthographic_camera(scene_name)
    
    def _render_emergence_cascade(self, cascade_data: Dict, row: int, col: int) -> None:
        """Render emergence cascade as interactive bar chart."""
        dimensions = cascade_data.get('dimensions', list(range(8)))
        densities = cascade_data.get('densities', np.random.rand(8))
        emergence_times = cascade_data.get('emergence_times', {})
        current_time = cascade_data.get('current_time', 0.0)
        
        # Color bars by emergence status
        colors = ['green' if dim in emergence_times else 'gray' for dim in dimensions]
        
        bars = go.Bar(
            x=dimensions,
            y=densities,
            marker_color=colors,
            name="Phase Densities",
            hovertemplate="<b>Dimension %{x}</b><br>Density: %{y:.3f}<extra></extra>"
        )
        self.fig.add_trace(bars, row=row, col=col)
        
        # Add emergence threshold line
        threshold = cascade_data.get('threshold', 0.5)
        threshold_line = go.Scatter(
            x=dimensions,
            y=[threshold] * len(dimensions),
            mode='lines',
            line=dict(dash='dash', color='red'),
            name="Emergence Threshold"
        )
        self.fig.add_trace(threshold_line, row=row, col=col)
    
    def _render_topology_view(self, topology_data: Dict, row: int, col: int) -> None:
        """Render topological structures with orthographic 3D."""
        topo_type = topology_data.get('type', 'torus')
        dimension = topology_data.get('dimension', 4.0)
        
        if topo_type == 'torus':
            self._render_torus_topology(topology_data, row, col, dimension)
        elif topo_type == 'gamma_surface':
            self._render_gamma_topology(topology_data, row, col, dimension)
        else:
            self._render_default_topology(topology_data, row, col, dimension)
            
        # Apply orthographic camera
        scene_name = f"scene{'' if row == 1 and col == 3 else str((row-1)*3 + col)}"
        self._apply_orthographic_camera(scene_name)
    
    def _render_torus_topology(self, data: Dict, row: int, col: int, dimension: float) -> None:
        """Render dimension-dependent torus."""
        u = np.linspace(0, 2*np.pi, 30)
        v = np.linspace(0, 2*np.pi, 30)
        u, v = np.meshgrid(u, v)
        
        # Dimension affects torus parameters
        R = 1.0 + 0.3 * np.sin(dimension)  # Major radius
        r = 0.3 + 0.1 * np.cos(dimension)  # Minor radius
        
        x = (R + r * np.cos(v)) * np.cos(u)
        y = (R + r * np.cos(v)) * np.sin(u)
        z = r * np.sin(v)
        
        torus = go.Surface(
            x=x, y=y, z=z,
            colorscale=self.color_sequences['topology'],
            opacity=0.8,
            name=f"Torus (d={dimension:.1f})"
        )
        self.fig.add_trace(torus, row=row, col=col)
    
    def _render_gamma_topology(self, data: Dict, row: int, col: int, dimension: float) -> None:
        """Render gamma function topology."""
        # Create gamma-based surface
        x = np.linspace(-2, 4, 40)
        y = np.linspace(-2, 4, 40)
        X, Y = np.meshgrid(x, y)
        
        # Safe gamma evaluation
        Z = np.zeros_like(X)
        for i in range(X.shape[0]):
            for j in range(X.shape[1]):
                try:
                    from scipy.special import gamma
                    val = X[i,j] + 1j*Y[i,j]
                    Z[i,j] = np.real(gamma(val + dimension/4))
                except:
                    Z[i,j] = 0
        
        # Clamp values for visualization
        Z = np.clip(Z, -10, 10)
        
        gamma_surface = go.Surface(
            x=X, y=Y, z=Z,
            colorscale=self.color_sequences['gamma'],
            opacity=0.9,
            name=f"Gamma Surface (d={dimension:.1f})"
        )
        self.fig.add_trace(gamma_surface, row=row, col=col)
    
    def _render_default_topology(self, data: Dict, row: int, col: int, dimension: float) -> None:
        """Render default topological view."""
        # Simple parametric surface
        u = np.linspace(0, 2*np.pi, 25)
        v = np.linspace(0, np.pi, 25)
        u, v = np.meshgrid(u, v)
        
        scale = dimension / 4.0
        x = scale * np.cos(u) * np.sin(v)
        y = scale * np.sin(u) * np.sin(v)  
        z = scale * np.cos(v)
        
        sphere = go.Surface(
            x=x, y=y, z=z,
            colorscale=self.color_sequences['topology'],
            opacity=0.7,
            name=f"Topological View (d={dimension:.1f})"
        )
        self.fig.add_trace(sphere, row=row, col=col)
    
    def _render_morphic_transformations(self, morphic_data: Dict, row: int, col: int) -> None:
        """Render morphic transformations with golden ratio."""
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        dimension = morphic_data.get('dimension', 4.0)
        
        t = np.linspace(0, 4*np.pi, 1000)
        scale = dimension / 4.0
        
        # Golden ratio spiral
        x = scale * np.cos(t) * (1 + 0.2 * np.cos(phi * t))
        y = scale * np.sin(t) * (1 + 0.2 * np.sin(phi * t))
        
        morphic_curve = go.Scatter(
            x=x, y=y,
            mode='lines',
            line=dict(width=3, color='gold'),
            name=f"Morphic Field (φ={phi:.3f})",
            hovertemplate="<b>Morphic Transform</b><br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>"
        )
        self.fig.add_trace(morphic_curve, row=row, col=col)
        
        # Add golden ratio points
        golden_points_t = np.array([0, np.pi/phi, 2*np.pi/phi, np.pi, 2*np.pi])
        golden_x = scale * np.cos(golden_points_t) * (1 + 0.2 * np.cos(phi * golden_points_t))
        golden_y = scale * np.sin(golden_points_t) * (1 + 0.2 * np.sin(phi * golden_points_t))
        
        golden_markers = go.Scatter(
            x=golden_x, y=golden_y,
            mode='markers',
            marker=dict(size=8, color='red', symbol='star'),
            name="Golden Ratio Points"
        )
        self.fig.add_trace(golden_markers, row=row, col=col)
    
    def _render_phase_flow(self, phase_data: Dict, row: int, col: int) -> None:
        """Render phase flow dynamics."""
        trajectories = phase_data.get('trajectories', [])
        
        for i, traj in enumerate(trajectories):
            time_points = traj.get('time', [])
            states = traj.get('states', [])
            
            if len(states) > 0 and len(states[0]) >= 2:
                x_vals = [state[0] for state in states]
                y_vals = [state[1] for state in states]
                
                traj_line = go.Scatter(
                    x=x_vals, y=y_vals,
                    mode='lines+markers',
                    line=dict(width=2),
                    name=f"Trajectory {i+1}",
                    hovertemplate="<b>Phase Trajectory</b><br>x: %{x:.3f}<br>y: %{y:.3f}<extra></extra>"
                )
                self.fig.add_trace(traj_line, row=row, col=col)
    
    def _render_controls_status(self, controls_data: Dict, row: int, col: int) -> None:
        """Render controls and status information."""
        # Create text display for controls
        status_text = controls_data.get('status_text', "System Status: Active")
        parameters = controls_data.get('parameters', {})
        
        # Create a simple text annotation plot
        dummy_scatter = go.Scatter(
            x=[0], y=[0],
            mode='markers',
            marker=dict(size=1, opacity=0),
            showlegend=False
        )
        self.fig.add_trace(dummy_scatter, row=row, col=col)
        
        # Add annotations for status
        subplot_ref = f"x{(row-1)*3 + col}" if (row-1)*3 + col > 1 else "x"
        yref = f"y{(row-1)*3 + col}" if (row-1)*3 + col > 1 else "y"
        
        self.fig.add_annotation(
            x=0.5, y=0.8,
            xref=subplot_ref, yref=yref,
            text=status_text,
            showarrow=False,
            font=dict(size=14, color="white"),
            xanchor="center"
        )
    
    def _render_unified_layout(self, geometry: Dict, topology: Dict, 
                             measures: Dict, parameters: Dict) -> None:
        """Render unified single-view layout."""
        # Render main component based on priority
        if 'landscape' in measures:
            self._render_unified_landscape(measures['landscape'])
        elif 'topology_view' in geometry:
            self._render_unified_topology(geometry['topology_view'])
    
    def _render_unified_landscape(self, landscape_data: Dict) -> None:
        """Render unified dimensional landscape."""
        d_range = landscape_data.get('d_range', np.linspace(0.1, 12, 100))
        volumes = landscape_data.get('volumes', [])
        surfaces = landscape_data.get('surfaces', [])
        complexity = landscape_data.get('complexity', [])
        
        # Volume curve
        vol_trace = go.Scatter(
            x=d_range[:len(volumes)],
            y=volumes,
            mode='lines',
            line=dict(width=3, color='blue'),
            name='Ball Volume'
        )
        self.fig.add_trace(vol_trace)
        
        # Surface curve
        surf_trace = go.Scatter(
            x=d_range[:len(surfaces)],
            y=surfaces,
            mode='lines', 
            line=dict(width=3, color='green'),
            name='Sphere Surface'
        )
        self.fig.add_trace(surf_trace)
        
        # Complexity curve
        if complexity:
            comp_trace = go.Scatter(
                x=d_range[:len(complexity)],
                y=complexity,
                mode='lines',
                line=dict(width=3, color='red'),
                name='Complexity Measure'
            )
            self.fig.add_trace(comp_trace)
    
    def _apply_orthographic_camera(self, scene_name: str) -> None:
        """Apply orthographic camera with mathematical constraints."""
        camera_dict = dict(
            eye=dict(x=1.25, y=1.25, z=1.25),  # Golden ratio positioning
            up=dict(x=0, y=0, z=1),
            projection=dict(type='orthographic')
        )
        
        scene_update = {
            f'{scene_name}': dict(
                camera=camera_dict,
                aspectmode='cube',  # 1:1:1 aspect ratio
                xaxis=dict(showgrid=True, zeroline=True, showticklabels=True),
                yaxis=dict(showgrid=True, zeroline=True, showticklabels=True),
                zaxis=dict(showgrid=True, zeroline=True, showticklabels=True)
            )
        }
        self.fig.update_layout(**scene_update)
    
    def update_camera(self, config: Optional[CameraConfig] = None) -> None:
        """Update orthographic camera maintaining mathematical constraints."""
        if config:
            self.camera = config
        
        # Apply to all 3D scenes
        for i in range(1, 7):  # Up to 6 subplots
            scene_key = f"scene{i}" if i > 1 else "scene"
            if hasattr(self.fig, 'layout') and hasattr(self.fig.layout, scene_key):
                self._apply_orthographic_camera(scene_key)
    
    def _apply_control_impl(self, control_type: str, value: Any) -> bool:
        """Apply Plotly-specific control operations."""
        try:
            if control_type == self.semantics.ADDITIVE:
                # Additive control affects plot ranges/scales
                if isinstance(value, (list, tuple, np.ndarray)) and len(value) >= 2:
                    x_range, y_range = value[0], value[1]
                    self.fig.update_xaxes(range=[x_range[0], x_range[1]])
                    self.fig.update_yaxes(range=[y_range[0], y_range[1]])
                    
            elif control_type == self.semantics.MULTIPLICATIVE:
                # Multiplicative control affects scaling/zoom
                scale_factor = float(value)
                current_ranges = self._get_current_ranges()
                for axis_update in current_ranges:
                    axis_update['range'] = [r * scale_factor for r in axis_update['range']]
                    
            elif control_type == self.semantics.BOUNDARY:
                # Boundary control affects visibility and styling
                visibility = bool(value)
                self.fig.update_traces(visible=visibility)
                    
            return True
            
        except Exception as e:
            print(f"❌ Plotly control application failed: {e}")
            return False
    
    def _get_current_ranges(self) -> List[Dict]:
        """Get current axis ranges for scaling operations."""
        ranges = []
        if hasattr(self.fig, 'layout'):
            for axis_name in ['xaxis', 'yaxis', 'zaxis']:
                axis = getattr(self.fig.layout, axis_name, None)
                if axis and hasattr(axis, 'range') and axis.range:
                    ranges.append({'name': axis_name, 'range': list(axis.range)})
        return ranges
    
    def export_scene(self, format_type: str = "html") -> Any:
        """Export scene in specified format."""
        if format_type == "html":
            return self.fig.to_html(
                include_plotlyjs='cdn',
                div_id="dimensional_dashboard",
                config={'displayModeBar': True, 'toImageButtonOptions': {'format': 'png'}}
            )
        elif format_type == "json":
            return self.fig.to_json()
        elif format_type == "image":
            # Requires kaleido: pip install kaleido
            return self.fig.to_image(format="png", width=1200, height=800)
        else:
            return self.fig
    
    def add_interactivity(self) -> None:
        """Add advanced Plotly interactivity features."""
        # Enable advanced interactions
        self.fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    direction="left",
                    buttons=list([
                        dict(args=[{"visible": [True] * len(self.fig.data)}], label="Show All", method="update"),
                        dict(args=[{"visible": [False] * len(self.fig.data)}], label="Hide All", method="update"),
                    ]),
                    pad={"r": 10, "t": 10},
                    showactive=True,
                    x=0.01,
                    xanchor="left",
                    y=1.02,
                    yanchor="top"
                ),
            ]
        )
        
        # Add range sliders where appropriate
        self.fig.update_layout(
            xaxis=dict(rangeslider=dict(visible=True)),
            hovermode='closest'
        )
    
    def create_mathematical_annotations(self, equations: List[str]) -> None:
        """Add LaTeX mathematical annotations."""
        for i, eq in enumerate(equations):
            self.fig.add_annotation(
                x=0.02, y=0.98 - i*0.05,
                xref="paper", yref="paper",
                text=f"${eq}$",  # LaTeX notation
                showarrow=False,
                font=dict(size=12, color="white"),
                bgcolor="rgba(0,0,0,0.5)",
                bordercolor="white",
                borderwidth=1
            )