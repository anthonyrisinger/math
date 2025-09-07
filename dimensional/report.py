"""
Beautiful HTML report generator for dimensional mathematics.
The killer feature that makes exploration results shareable.
"""

from datetime import datetime

import numpy as np

from .core import c, gamma, r, s, v


def generate_report(d=None, filename="dimensional_report.html"):
    """
    Generate a beautiful HTML report of dimensional analysis.

    Args:
        d: Dimension to analyze (default: finds optimal)
        filename: Output HTML filename

    Returns:
        Path to generated HTML file
    """
    if d is None:
        # Find the most interesting dimension (complexity peak)
        from scipy.optimize import minimize_scalar
        result = minimize_scalar(lambda x: -c(x), bounds=(1, 15), method='bounded')
        d = result.x

    # Calculate all measures
    volume = v(d)
    surface = s(d)
    complexity = c(d)
    ratio = r(d)
    gamma_val = gamma(d/2) if d > 0 else None

    # Generate dimension range for plots
    d_range = np.linspace(max(0.1, d-5), min(d+5, 30), 200)
    v_vals = v(d_range)
    s_vals = s(d_range)
    c_vals = c(d_range)

    # Find peaks
    v_peak_idx = np.argmax(v_vals)
    s_peak_idx = np.argmax(s_vals)
    c_peak_idx = np.argmax(c_vals)

    # Create beautiful HTML
    html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dimensional Analysis Report - d={d:.3f}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 2rem;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: rgba(255, 255, 255, 0.95);
            border-radius: 20px;
            padding: 3rem;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
        }}
        h1 {{
            color: #2d3748;
            font-size: 2.5rem;
            margin-bottom: 0.5rem;
            text-align: center;
        }}
        .subtitle {{
            text-align: center;
            color: #718096;
            margin-bottom: 2rem;
        }}
        .metrics {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.5rem;
            margin: 2rem 0;
        }}
        .metric {{
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
            text-align: center;
            transform: translateY(0);
            transition: transform 0.3s;
        }}
        .metric:hover {{
            transform: translateY(-5px);
        }}
        .metric:nth-child(2) {{
            background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        }}
        .metric:nth-child(3) {{
            background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        }}
        .metric:nth-child(4) {{
            background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
        }}
        .metric-label {{
            font-size: 0.9rem;
            opacity: 0.9;
            margin-bottom: 0.5rem;
        }}
        .metric-value {{
            font-size: 1.8rem;
            font-weight: bold;
        }}
        .plot {{
            margin: 2rem 0;
            border-radius: 10px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }}
        .insights {{
            background: #f7fafc;
            padding: 2rem;
            border-radius: 10px;
            margin: 2rem 0;
        }}
        .insights h2 {{
            color: #2d3748;
            margin-bottom: 1rem;
        }}
        .insight {{
            padding: 0.5rem 0;
            color: #4a5568;
        }}
        .footer {{
            text-align: center;
            margin-top: 3rem;
            padding-top: 2rem;
            border-top: 1px solid #e2e8f0;
            color: #718096;
        }}
        .special {{
            background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Dimensional Analysis Report</h1>
        <p class="subtitle">Exploring dimension <span class="special">d = {d:.6f}</span></p>

        <div class="metrics">
            <div class="metric">
                <div class="metric-label">Volume V(d)</div>
                <div class="metric-value">{volume:.4e}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Surface S(d)</div>
                <div class="metric-value">{surface:.4e}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Complexity C(d)</div>
                <div class="metric-value">{complexity:.4e}</div>
            </div>
            <div class="metric">
                <div class="metric-label">Ratio R(d)</div>
                <div class="metric-value">{ratio:.4f}</div>
            </div>
        </div>

        <div id="plot" class="plot"></div>

        <div class="insights">
            <h2>🔍 Key Insights</h2>
            <div class="insight">📊 Volume peaks at dimension <strong>{d_range[v_peak_idx]:.3f}</strong> with value {v_vals[v_peak_idx]:.4e}</div>
            <div class="insight">🌐 Surface peaks at dimension <strong>{d_range[s_peak_idx]:.3f}</strong> with value {s_vals[s_peak_idx]:.4e}</div>
            <div class="insight">🎯 Complexity peaks at dimension <strong>{d_range[c_peak_idx]:.3f}</strong> with value {c_vals[c_peak_idx]:.4e}</div>
            <div class="insight">✨ The ratio S/V = {ratio:.4f} indicates {'high' if ratio > 5 else 'moderate' if ratio > 2 else 'low'} surface density</div>
            {f'<div class="insight">🔢 Gamma function Γ(d/2) = {gamma_val:.4e}</div>' if gamma_val else ''}
        </div>

        <div class="footer">
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} by Dimensional Mathematics Framework v3.0</p>
            <p style="margin-top: 1rem;">✨ <em>Beauty in Higher Dimensions</em> ✨</p>
        </div>
    </div>

    <script>
        var trace1 = {{
            x: {d_range.tolist()},
            y: {v_vals.tolist()},
            mode: 'lines',
            name: 'Volume',
            line: {{color: '#f093fb', width: 3}}
        }};

        var trace2 = {{
            x: {d_range.tolist()},
            y: {s_vals.tolist()},
            mode: 'lines',
            name: 'Surface',
            line: {{color: '#4facfe', width: 3}}
        }};

        var trace3 = {{
            x: {d_range.tolist()},
            y: {c_vals.tolist()},
            mode: 'lines',
            name: 'Complexity',
            line: {{color: '#43e97b', width: 3}},
            yaxis: 'y2'
        }};

        // Mark current dimension
        var current = {{
            x: [{d}],
            y: [{volume}],
            mode: 'markers',
            name: 'Current',
            marker: {{size: 15, color: '#f5576c', symbol: 'star'}},
            showlegend: false
        }};

        var layout = {{
            title: 'Dimensional Measures Near d = {d:.3f}',
            xaxis: {{title: 'Dimension', gridcolor: '#e2e8f0'}},
            yaxis: {{title: 'Volume / Surface', type: 'log', gridcolor: '#e2e8f0'}},
            yaxis2: {{
                title: 'Complexity',
                overlaying: 'y',
                side: 'right',
                type: 'log',
                gridcolor: '#e2e8f0'
            }},
            paper_bgcolor: 'rgba(0,0,0,0)',
            plot_bgcolor: 'white',
            hovermode: 'x unified',
            legend: {{x: 0.02, y: 0.98}},
            margin: {{t: 50}}
        }};

        Plotly.newPlot('plot', [trace1, trace2, trace3, current], layout, {{responsive: true}});
    </script>
</body>
</html>
"""

    # Write the HTML file
    with open(filename, 'w') as f:
        f.write(html)

    print(f"✨ Beautiful report generated: {filename}")
    print(f"📊 Analyzed dimension {d:.6f}")
    print(f"🎯 Complexity peak found at {d_range[c_peak_idx]:.3f}")

    return filename
