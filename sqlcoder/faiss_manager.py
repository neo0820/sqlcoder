import faiss
import numpy as np
import os

class FaissManager:
    def __init__(self, base_dir: str, dim: int = 768):
        """
        初始化FAISS管理类。
        :param base_dir: 索引文件存储的基础目录。
        :param dim: 向量维度，默认768。
        """
        self.base_dir = base_dir
        self.dim = dim
        self.index_map = {}  # 缓存已加载的索引
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)

    def _get_index_path(self, db_name: str) -> str:
        """
        获取索引文件路径。
        :param db_name: 数据库名，作为索引文件的标识。
        :return: 索引文件路径。
        """
        return os.path.join(self.base_dir, f"{db_name}_index.bin")

    def load_or_create_index(self, db_name: str):
        """
        加载索引，如果不存在则新建。
        :param db_name: 数据库名。
        """
        index_path = self._get_index_path(db_name)
        if os.path.exists(index_path):
            print(f"加载现有索引: {index_path}")
            index = faiss.read_index(index_path)
        else:
            print(f"索引不存在，创建新索引: {db_name}")
            index = faiss.IndexFlatL2(self.dim)  # 初始化新索引
        self.index_map[db_name] = index

    def add_vectors(self, db_name: str, vectors: np.ndarray):
        """
        添加向量到指定数据库的索引中。
        :param db_name: 数据库名。
        :param vectors: 新的向量，必须为 numpy.ndarray，且 dtype 为 float32。
        :return: 向量的主键ID（即索引ID）。
        """
        if db_name not in self.index_map:
            self.load_or_create_index(db_name)

        index = self.index_map[db_name]
        
        if vectors.shape[1] != self.dim:
            raise ValueError(f"向量维度不匹配！索引需要 {self.dim} 维，实际为 {vectors.shape[1]} 维。")

        # 向量的ID会自动生成，ID从0开始递增
        index.add(vectors)
        print(f"添加 {vectors.shape[0]} 个向量到索引: {db_name}")

        # 返回新增的向量的ID
        return np.arange(index.ntotal - vectors.shape[0], index.ntotal)

    def save_index(self, db_name: str):
        """
        保存指定数据库的索引到磁盘。
        :param db_name: 数据库名。
        """
        if db_name in self.index_map:
            index_path = self._get_index_path(db_name)
            faiss.write_index(self.index_map[db_name], index_path)
            print(f"索引已保存: {index_path}")
        else:
            print(f"未找到数据库 {db_name} 的索引，无法保存。")

    def search(self, db_name: str, query_vectors: np.ndarray, top_k: int = 5):
        """
        在指定数据库的索引中搜索最接近的向量。
        :param db_name: 数据库名。
        :param query_vectors: 查询向量。
        :param top_k: 返回的最近邻数量，默认5。
        :return: 符合条件的向量主键ID（即FAISS中的ID）集合。 最近邻的距离和索引 (distances, indices)。
        """
        if db_name not in self.index_map:
            self.load_or_create_index(db_name)

        index = self.index_map[db_name]

        if query_vectors.shape[1] != self.dim:
            raise ValueError(f"查询向量维度不匹配！索引需要 {self.dim} 维，实际为 {query_vectors.shape[1]} 维。")

        distances, indices = index.search(query_vectors, top_k)

        # 返回符合条件的索引ID集合
        return indices