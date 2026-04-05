# Detection Dataclass

Defined in `ms/ObjectDetection/detection.py`. Replaces the implicit dict schema with a typed, slotted dataclass while preserving full dict-style access for backward compatibility.

```python
from ms.ObjectDetection.detection import Detection
```

---

## Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `class_name` | str | (required) | YOLO class name (e.g. `"person"`, `"chair"`) |
| `cls_id` | int | (required) | YOLO class index |
| `conf` | float | (required) | Detection confidence score (0.0--1.0) |
| `x1` | int | (required) | Bounding box left |
| `y1` | int | (required) | Bounding box top |
| `x2` | int | (required) | Bounding box right |
| `y2` | int | (required) | Bounding box bottom |
| `ghost` | bool | `False` | Whether this is a persisted (ghost) detection from `ObjectPersistenceCache` |
| `_face_idx` | int or None | `None` | Internal index used when faces are treated as objects (not shown in `repr`) |

---

## Dict-Compatible API

Existing code that treated detections as dicts continues to work unchanged.

### `__getitem__(key)`

```python
det = Detection(class_name="person", cls_id=0, conf=0.92, x1=10, y1=20, x2=100, y2=200)
det['x1']          # 10
det['_ghost']       # False  (aliased from 'ghost' via _KEY_MAP)
```

### `get(key, default=None)`

```python
det.get('_face_idx')       # None
det.get('missing', -1)     # -1
```

### `__contains__(key)`

```python
'conf' in det        # True
'_ghost' in det      # True (resolves alias)
'nonexist' in det    # False
```

### `__setitem__(key, value)`

```python
det['ghost'] = True
det['_ghost'] = True   # equivalent (alias resolved)
```

### `update(**kwargs)`

```python
det.update(ghost=True, conf=0.5)
```

### `keys()`, `values()`, `items()`

These allow iteration and unpacking, matching dict behaviour. `keys()` returns field names including `_face_idx` but excluding other private fields.

```python
list(det.keys())    # ['class_name', 'cls_id', 'conf', 'x1', 'y1', 'x2', 'y2', 'ghost', '_face_idx']
dict(det.items())   # full dict representation
```

---

## Key Alias Map

The `_KEY_MAP` class variable maps legacy dict keys to dataclass field names:

| Dict Key | Field Name |
|----------|-----------|
| `_ghost` | `ghost` |

All other keys map to themselves.

---

## Properties

### `center`

Returns the bounding-box centre as a float numpy array `[cx, cy]`.

```python
det.center   # np.array([55.0, 110.0])
```

---

## Notes

- The dataclass uses `slots=True` for memory efficiency.
- `Detection` supports `__iter__`, which yields keys -- so `dict(det, _ghost=True)` patterns work.
- Ghost detections are created by `ObjectPersistenceCache` when a previously-seen object is not detected for up to `obj_persistence` frames.
