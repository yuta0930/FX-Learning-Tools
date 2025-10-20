import importlib
import sys
import types


class _DummyCtx:
    def __enter__(self):
        return self
    def __exit__(self, exc_type, exc, tb):
        return False


class _Session:
    def __init__(self):
        self._d = {}
    def setdefault(self, k, v):
        return self._d.setdefault(k, v)
    def get(self, k, d=None):
        return self._d.get(k, d)
    def __contains__(self, k):
        return k in self._d
    def __getitem__(self, k):
        return self._d[k]
    def __setitem__(self, k, v):
        self._d[k] = v
    def __getattr__(self, k):
        try:
            return self._d[k]
        except KeyError:
            raise AttributeError(k)
    def __setattr__(self, k, v):
        if k == '_d':
            object.__setattr__(self, k, v)
        else:
            self._d[k] = v


def _install_fake_streamlit_modules():
    st = types.ModuleType('streamlit')
    ss = _Session()
    st.session_state = ss

    # common no-op UI functions
    def _noop(*a, **kw):
        return None
    def _toggle(*a, **kw):
        return False
    class _Column:
        def __init__(self, st_mod):
            self._st = st_mod
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
        def __getattr__(self, name):
            # Delegate widget/layout calls to top-level st functions
            return getattr(self._st, name)
    def _columns(spec):
        n = 2
        if isinstance(spec, int):
            n = max(1, spec)
        elif isinstance(spec, (list, tuple)):
            n = max(1, len(spec))
        return tuple(_Column(st) for _ in range(n))
    def _expander(*a, **kw):
        return _DummyCtx()
    def _tabs(labels):
        return [ _DummyCtx() for _ in (labels or []) ]
    def _cache_like(fn=None, *a, **kw):
        if callable(fn):
            return fn
        def deco(f):
            return f
        return deco
    def _selectbox(label, options, index=0, *a, **kw):
        try:
            return options[index]
        except Exception:
            return options[0] if options else None
    def _number_input(label, value=0, *a, **kw):
        return value
    def _slider(label, min_value=None, max_value=None, value=None, step=None, *a, **kw):
        return value if value is not None else min_value
    def _checkbox(label, value=False, *a, **kw):
        return value
    def _text_area(label, value="", *a, **kw):
        return value or ""
    def _radio(label, options, index=0, *a, **kw):
        try:
            return options[index]
        except Exception:
            return options[0] if options else None
    def _file_uploader(label, type=None, *a, **kw):
        return None
    def _multiselect(label, options=None, default=None, *a, **kw):
        return default or []
    def _date_input(label, value=None, *a, **kw):
        return value
    def _download_button(label, data=None, file_name=None, mime=None, *a, **kw):
        return False
    def _button(*a, **kw):
        return False
    class _Spinner(_DummyCtx):
        pass
    def _spinner(*a, **kw):
        return _Spinner()
    def _stop():
        raise SystemExit("st.stop called")

    # assign
    st.set_page_config = _noop
    st.columns = _columns
    st.expander = _expander
    st.container = lambda *a, **kw: _DummyCtx()
    st.tabs = _tabs
    st.popover = lambda *a, **kw: _DummyCtx()
    st.toggle = _toggle
    st.metric = _noop
    st.markdown = _noop
    st.header = _noop
    st.subheader = _noop
    st.json = _noop
    st.table = _noop
    st.dataframe = _noop
    st.image = _noop
    st.write = _noop
    st.warning = _noop
    st.info = _noop
    st.error = _noop
    st.caption = _noop
    st.line_chart = _noop
    st.plotly_chart = _noop
    st.text_area = _text_area
    st.radio = _radio
    st.file_uploader = _file_uploader
    st.multiselect = _multiselect
    st.date_input = _date_input
    st.download_button = _download_button
    st.button = _button
    st.selectbox = _selectbox
    st.number_input = _number_input
    st.slider = _slider
    st.checkbox = _checkbox
    st.spinner = _spinner
    st.empty = lambda: None
    st.stop = _stop
    st.toast = _noop
    st.cache_resource = _cache_like
    st.cache_data = _cache_like

    # sidebar with same API
    class _Sidebar:
        def __init__(self, st_mod):
            self._st = st_mod
            # expose common methods directly
            self.title = _noop
            self.header = _noop
            self.subheader = _noop
            self.text_input = lambda label, value="", *a, **kw: value
            self.selectbox = _selectbox
            self.expander = _expander
            self.number_input = _number_input
            self.checkbox = _checkbox
            self.slider = _slider
            self.toggle = _toggle
            self.markdown = _noop
            self.caption = _noop
            self.code = _noop
            self.button = _button
            self.download_button = _download_button
            self.multiselect = _multiselect
            self.date_input = _date_input
        def __getattr__(self, name):
            # delegate unknown attributes to top-level st
            return getattr(self._st, name)
    st.sidebar = _Sidebar(st)

    # components submodules used by streamlit_autorefresh
    comp = types.ModuleType('streamlit.components')
    comp_v1 = types.ModuleType('streamlit.components.v1')
    comp_v1.html = _noop

    sys.modules['streamlit'] = st
    sys.modules['streamlit.components'] = comp
    sys.modules['streamlit.components.v1'] = comp_v1


def test_app_import_smoke():
    _install_fake_streamlit_modules()
    # 既に読み込まれていたら消して再インポート
    if 'app' in sys.modules:
        del sys.modules['app']
    try:
        importlib.import_module('app')
    except Exception as e:
        raise AssertionError(f"app import failed: {e}")
