import time
from typing import Iterable, TypeVar, Optional, Union, Any

T = TypeVar('T')

class StreamlitProgress:
    """tqdmと同じインターフェースを持つStreamlitのプログレスバー"""

    def __init__(self, iterable: Optional[Iterable[T]] = None, total: Optional[int] = None,
                 desc: str = "", **kwargs):
        self.iterable = iterable
        self.total = total if total is not None else (len(iterable) if iterable is not None and hasattr(iterable, "__len__") else None)
        self.desc = desc
        self.n = 0
        self.kwargs = kwargs

        try:
            import streamlit as st
            # 説明テキスト用のコンテナ
            self.text_container = st.empty()
            if desc:
                self.text_container.text(desc)
            # プログレスバー
            self.progress_bar = st.progress(0)
            # 後置テキスト用のコンテナ
            self.postfix_container = st.empty()
        except ImportError:
            raise ImportError("streamlitがインストールされていません。")

    def update(self, n: int = 1) -> None:
        """進捗を更新する"""
        self.n += n
        if self.total:
            self.progress_bar.progress(min(self.n / self.total, 1.0))

    def set_description(self, desc: str = None, refresh: bool = True) -> None:
        """説明テキストを更新する"""
        if desc is not None:
            self.desc = desc
            # self.text_container.text(desc)
            self.text_container.markdown('<p style="font-size:14px; color:gray;">' + desc +'</p>', unsafe_allow_html=True)


    def set_postfix(self, ordered_dict=None, **kwargs) -> None:
        """後置テキストを設定する"""
        # ordered_dictとkwargsを組み合わせる
        postfix_dict = {}
        if ordered_dict:
            postfix_dict.update(ordered_dict)
        if kwargs:
            postfix_dict.update(kwargs)

        if postfix_dict:
            # 辞書を文字列に変換して表示
            postfix_str = ', '.join(f'{k}={v}' for k, v in postfix_dict.items())
            self.postfix_container.text(f"状態: {postfix_str}")

    def close(self) -> None:
        """プログレスバーを完了状態にする"""
        if self.total:
            self.progress_bar.progress(1.0)
        self.text_container.empty()

    def refresh(self):
        """ 不要なので何もしない """
        pass

    def __iter__(self):
        """イテレータとして使用できるようにする"""
        if self.iterable is None:
            raise ValueError("このプログレスバーはイテレータとして使用できません")

        for obj in self.iterable:
            yield obj
            self.update(1)

        self.close()

    def __enter__(self):
        """コンテキストマネージャとして使用できるようにする"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """コンテキスト終了時に呼ばれる"""
        self.close()

def tqdm_or_st(iterable: Optional[Iterable[T]] = None,
              backend: str = 'tqdm',
              **kwargs) -> Union['tqdm', StreamlitProgress]:
    """
    指定されたバックエンドのプログレスバーを返す

    Args:
        iterable: 進捗を表示するイテレータ
        backend: バックエンド ("tqdm", "streamlit")
        **kwargs: tqdmやStreamlitProgressに渡す引数

    Returns:
        tqdm または StreamlitProgress オブジェクト
    """
    # if backend == "auto":
    #     try:
    #         import streamlit as st
    #         if st._is_running_with_streamlit:
    #             backend = "streamlit"
    #         else:
    #             backend = "tqdm"
    #     except (ImportError, AttributeError):
    #         backend = "tqdm"

    assert backend in ['tqdm', 'streamlit']

    if backend == "tqdm":
        try:
            from tqdm import tqdm
            return tqdm(iterable, **kwargs)
        except ImportError:
            print("tqdmが見つからないため、Streamlitバックエンドを試行します...")
            backend = "streamlit"

    # Streamlitを使用
    if backend == "streamlit":
        try:
            return StreamlitProgress(iterable, **kwargs)
        except ImportError:
            print("Streamlitが見つかりません。プログレスバーなしで実行します。")
            # フォールバック: 何もしないダミープログレスバー
            try:
                from tqdm import tqdm
                return tqdm(iterable, disable=True, **kwargs)
            except ImportError:
                # tqdmもないので、単なるイテレータを返す
                class DummyTqdm:
                    def __init__(self, iterable=None, **kwargs):
                        self.iterable = iterable
                    def update(self, n=1): pass
                    def close(self): pass
                    def set_description(self, desc=None, refresh=True): pass
                    def set_postfix(self, ordered_dict=None, **kwargs): pass
                    def __iter__(self):
                        if self.iterable is None: raise ValueError("イテレータがありません")
                        for x in self.iterable: yield x
                    def __enter__(self): return self
                    def __exit__(self, *args, **kwargs): pass
                return DummyTqdm(iterable, **kwargs)

# 基本的な使用例
def basic_example():
    """基本的な使用例"""
    items = list(range(10))

    # tqdmと同じ使い方
    for item in tqdm_or_st(items, desc="基本的な例", backend="tqdm"):
        time.sleep(0.1)
        print(f"処理中: {item}")

# Streamlitの使用例
def streamlit_example():
    """Streamlitでの使用例 (Streamlitアプリ内で実行する必要があります)"""
    import streamlit as st

    st.title("処理の進捗表示")

    items = list(range(10))
    results = []

    # 自動的にStreamlitを検出
    for item in tqdm_or_st(items, desc="処理中...", backend="auto"):
        time.sleep(0.2)
        results.append(item * 2)

    st.write("結果:", results)

# コンテキストマネージャとしての使用例
def context_manager_example():
    """コンテキストマネージャとしての使用例"""
    total_steps = 5

    # with文で使用
    with tqdm_or_st(total=total_steps, desc="手動更新", backend="tqdm") as pbar:
        for i in range(total_steps):
            time.sleep(0.2)

            # 説明を更新
            if i == 2:
                pbar.set_description(f"ステップ {i+1}/{total_steps}")

            # 追加情報を表示
            pbar.set_postfix(progress=f"{(i+1)/total_steps:.0%}")

            # 進捗を更新
            pbar.update(1)

# テスト用のメイン関数
def main():
    print("基本的な使用例:")
    basic_example()

    print("\nコンテキストマネージャとしての使用例:")
    context_manager_example()

    print("\nStreamlitの例はStreamlitアプリ内で実行してください")
    # streamlit_example()  # Streamlitアプリ内でのみ実行可能

if __name__ == "__main__":
    main()
