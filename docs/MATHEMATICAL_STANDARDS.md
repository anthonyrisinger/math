# Mathematical Standards

For contributors implementing mathematical functions in the core library.

## Function Documentation

```python
def sphere_volume(d):
    """
    Volume of unit d-sphere.
    
    Formula: V_d = π^(d/2) / Γ(d/2 + 1)
    
    Args:
        d (float): Dimension (d > 0)
        
    Returns:
        float: Volume of unit d-sphere
        
    Examples:
        >>> sphere_volume(2)  # Unit circle area
        3.141592653589793
        >>> sphere_volume(3)  # Unit sphere volume  
        4.188790204786391
    """
```

## Testing Requirements

Every mathematical function needs:

```python
def test_sphere_volume_exact_values():
    """Test against known exact values."""
    assert abs(V(2) - np.pi) < 1e-12
    assert abs(V(3) - 4*np.pi/3) < 1e-12

def test_sphere_volume_properties():
    """Test mathematical properties."""
    # Test recurrence relation: V(d) = V(d-2) * 2π/d
    for d in range(3, 10):
        expected = V(d-2) * 2 * np.pi / d
        assert abs(V(d) - expected) / V(d) < 1e-12
```

## Numerical Stability

For large dimensions, use logarithmic forms:

```python
def log_sphere_volume(d):
    """Compute log of sphere volume for numerical stability."""
    return (d/2) * np.log(np.pi) - gammaln(d/2 + 1)
```

## Edge Cases

Handle special values explicitly:

```python
def gamma_function(z):
    # Handle exact values
    if z == 1.0 or z == 2.0:
        return 1.0
    # Handle poles  
    if z <= 0 and z == int(z):
        raise ValueError(f"Gamma has pole at z={z}")
```

That's it. Keep the math correct and well-tested.