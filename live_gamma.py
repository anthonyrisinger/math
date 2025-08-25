#!/usr/bin/env python3
"""
LIVE GAMMA - Hot-reload exploration system
===========================================
Watches gamma_expr.py and auto-refreshes plots on save.
Edit gamma_expr.py in your editor, save, and see instant results!
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, gammaln, digamma
import os
import time
import importlib.util
import traceback
from pathlib import Path

# Constants
π = np.pi
φ = (1 + np.sqrt(5))/2

EXPR_FILE = "gamma_expr.py"
TEMPLATE = '''#!/usr/bin/env python3
"""
Edit this file and save to see live updates!
Define a function called plot(fig, d) that creates your visualization.
The parameter d is the current dimension value.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma, gammaln, digamma, polygamma

π = np.pi
φ = (1 + np.sqrt(5))/2

def plot(fig, d=4.0):
    """Your visualization here - edit and save!"""
    
    # Example: V×S complexity around dimension d
    fig.clear()
    ax = fig.add_subplot(111, facecolor='#1a1a1a')
    
    # Define measures
    v = lambda x: π**(x/2) / gamma(x/2 + 1)
    s = lambda x: 2*π**(x/2) / gamma(x/2)
    c = lambda x: v(x) * s(x)
    
    # Plot range around current d
    x = np.linspace(max(0.1, d-3), d+3, 500)
    
    # Plot complexity
    y = [c(xi) for xi in x]
    ax.plot(x, y, 'lime', lw=3, alpha=0.8)
    
    # Mark current point
    ax.axvline(d, color='yellow', lw=2, alpha=0.5)
    ax.plot(d, c(d), 'yo', markersize=10)
    ax.text(d, c(d), f'  d={d:.2f}, C={c(d):.3f}', color='yellow')
    
    # Style
    ax.set_title(f'Complexity C = V×S around d={d:.2f}', color='white', fontsize=14)
    ax.set_xlabel('Dimension', color='white')
    ax.set_ylabel('V×S', color='white')
    ax.grid(True, alpha=0.2)
    
    # Try different things!
    # - Plot gamma directly: y = [gamma(xi) for xi in x]
    # - Show poles: x = np.linspace(-5, 5, 1000)
    # - Complex plane: use meshgrid and contourf
    # - Multiple subplots: use fig.add_subplot(2,2,1) etc
'''

class LiveGamma:
    def __init__(self):
        self.d = 4.0  # Current dimension
        self.last_mtime = 0
        self.module = None
        
        # Create expression file if it doesn't exist
        if not os.path.exists(EXPR_FILE):
            with open(EXPR_FILE, 'w') as f:
                f.write(TEMPLATE)
            print(f"✨ Created {EXPR_FILE} - edit it and save to see live updates!")
        
        # Setup figure
        self.fig = plt.figure(figsize=(12, 8), facecolor='#0a0a0a')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        # Initial load
        self.reload_module()
        
        # Start watching
        self.watch()
    
    def on_key(self, event):
        """Handle keyboard input"""
        if event.key == 'up':
            self.d += 0.1
            self.update_plot()
        elif event.key == 'down':
            self.d -= 0.1
            self.update_plot()
        elif event.key == 'right':
            self.d += 1.0
            self.update_plot()
        elif event.key == 'left':
            self.d -= 1.0
            self.update_plot()
        elif event.key == 'r':
            self.d = 4.0
            self.update_plot()
        elif event.key == 'space':
            self.reload_module()
        elif event.key in ['q', 'escape']:
            plt.close('all')
            exit()
    
    def reload_module(self):
        """Reload the expression module"""
        try:
            # Load or reload module
            spec = importlib.util.spec_from_file_location("gamma_expr", EXPR_FILE)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Check if plot function exists
            if not hasattr(module, 'plot'):
                self.show_error("No plot() function found in gamma_expr.py")
                return
            
            self.module = module
            self.update_plot()
            
        except Exception as e:
            self.show_error(f"Error loading module:\n{traceback.format_exc()}")
    
    def update_plot(self):
        """Update the plot with current module"""
        if self.module is None:
            return
        
        try:
            # Clear and set title
            self.fig.clear()
            self.fig.text(0.5, 0.98, f'LIVE GAMMA | d = {self.d:.3f} | ↑↓←→: change d | SPACE: reload | Q: quit', 
                         ha='center', fontsize=12, color='white', weight='bold')
            
            # Call user's plot function
            self.module.plot(self.fig, self.d)
            
            # Refresh
            self.fig.canvas.draw_idle()
            
        except Exception as e:
            self.show_error(f"Error in plot():\n{traceback.format_exc()}")
    
    def show_error(self, msg):
        """Display error message"""
        self.fig.clear()
        ax = self.fig.add_subplot(111, facecolor='#1a1a1a')
        ax.text(0.5, 0.5, msg, ha='center', va='center', 
                color='red', fontsize=10, family='monospace',
                transform=ax.transAxes)
        ax.set_xticks([])
        ax.set_yticks([])
        self.fig.canvas.draw_idle()
    
    def watch(self):
        """Watch for file changes"""
        print(f"\n🔥 LIVE MODE ACTIVE")
        print(f"   Watching: {EXPR_FILE}")
        print(f"   Edit the file and save to see changes!")
        print(f"\n   Keyboard controls:")
        print(f"   ↑/↓     : d ± 0.1")
        print(f"   ←/→     : d ± 1.0")
        print(f"   R       : Reset to d=4")
        print(f"   SPACE   : Force reload")
        print(f"   Q/ESC   : Quit\n")
        
        while plt.fignum_exists(self.fig.number):
            try:
                # Check file modification time
                current_mtime = os.path.getmtime(EXPR_FILE)
                if current_mtime > self.last_mtime:
                    self.last_mtime = current_mtime
                    print(f"📝 Detected change, reloading...")
                    self.reload_module()
                
                # Small pause to not hog CPU
                plt.pause(0.1)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Watch error: {e}")
                time.sleep(1)

def live_snippet(code_str):
    """Quick live evaluation of code snippet"""
    fig = plt.figure(figsize=(10, 6), facecolor='#0a0a0a')
    
    # Create namespace with common imports
    namespace = {
        'np': np,
        'plt': plt,
        'gamma': gamma,
        'gammaln': gammaln,
        'digamma': digamma,
        'π': π,
        'φ': φ,
        'fig': fig,
        'd': 4.0
    }
    
    # Execute code
    try:
        exec(code_str, namespace)
        plt.show()
    except Exception as e:
        print(f"Error: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    print(__doc__)
    
    # Check if running with snippet
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--snippet':
        if len(sys.argv) > 2:
            code = ' '.join(sys.argv[2:])
            live_snippet(code)
        else:
            print("Usage: python live_gamma.py --snippet 'your code here'")
    else:
        # Run live mode
        live = LiveGamma()
        plt.show()