"""
Streamlit session state management
"""

import streamlit as st


def add_beforeunload_js():
    """ページ離脱時の警告JSを追加"""
    js = """
    <script>
        window.onbeforeunload = function(e) {
            if (window.localStorage.getItem('streamlit_locked') === 'true') {
                e.preventDefault();
                e.returnValue = "処理中にページを離れると処理がリセットされます。ページを離れますか？";
                return e.returnValue;
            }
        };
    </script>
    """
    st.components.v1.html(js, height=0)


def set_locked_state(is_locked: bool):
    """ロック状態を設定"""
    print("locked", is_locked)
    st.session_state.locked = is_locked
    js = f"""
    <script>
        window.localStorage.setItem('streamlit_locked', '{str(is_locked).lower()}');
    </script>
    """
    st.components.v1.html(js, height=0)


def lock():
    """処理中ロックを有効化"""
    set_locked_state(True)


def unlock():
    """処理中ロックを解除"""
    set_locked_state(False)
    # キャッシュをクリア（処理後にファイルが更新されているため）
    st.cache_data.clear()


def render_reset_button():
    """リセットボタンを表示"""
    if st.button("リセットする", on_click=unlock):
        st.rerun()
