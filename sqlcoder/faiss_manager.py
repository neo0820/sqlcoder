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
            #采用算法：欧氏距离（L2 距离）
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

        # 改为手动计算新增的向量ID
        start_id = index.ntotal - vectors.shape[0]
        end_id = index.ntotal
        vector_ids = np.arange(start_id, end_id)

        # 保存索引到磁盘
        self.save_index(db_name)

        # 返回新增的向量的ID
        return vector_ids


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

        # 打印向量维度和类型以进行调试
        print("Adding vectors with shape:", query_vectors.shape)
        print("Data type:", query_vectors.dtype)
        
        print("==============~~~~~~~~~vectors shape:", query_vectors.shape)
        # 假设 vectors 是一维数组 
        print("vectors shape before reshape:", query_vectors.shape)
        if len(query_vectors.shape) == 1:
            query_vectors = query_vectors.reshape(1, -1)  # 转换成二维形状，(1, 768)
            print("=====是一维数组======")
        else:
            print("=====是二维数组======")
        print("vectors shape after reshape:", query_vectors.shape)


        if query_vectors.shape[1] != self.dim:
            raise ValueError(f"查询向量维度不匹配！索引需要 {self.dim} 维，实际为 {query_vectors.shape[1]} 维。")

        distances, indices = index.search(query_vectors, top_k)
        #indices返回的类似于这样的二维数组，[[ 0  1 -1 -1 -1]]
        # 返回符合条件的索引ID集合
        value= {"indices":indices.tolist()[0],"distances":distances.tolist()[0]}
        return value
    def delete_vectors(self, db_name: str, vector_ids: np.ndarray):
        """
        从索引中删除指定的向量。
        :param db_name: 数据库名。
        :param vector_ids: 需要删除的向量ID数组。
        """
        if db_name not in self.index_map:
            raise ValueError(f"数据库 {db_name} 的索引未加载，无法删除向量。")

        index = self.index_map[db_name]

        # 判断索引类型，确保支持删除操作
        if not isinstance(index, faiss.IndexIVF):
            raise NotImplementedError("当前索引类型不支持删除操作，请使用 IVF 索引。")

        # 删除指定的向量
        index.remove_ids(faiss.IDSelectorBatch(vector_ids))
        print(f"从索引 {db_name} 删除了 {len(vector_ids)} 个向量。")

        # 保存索引到磁盘
        self.save_index(db_name)

        
    def delete_index(self, db_name: str):
        """
        删除指定数据库的索引文件和缓存的索引。
        :param db_name: 数据库名。
        """
        if db_name in self.index_map:
            del self.index_map[db_name]  # 从内存缓存中移除
            print(f"已从缓存中移除索引: {db_name}")

        index_path = self._get_index_path(db_name)
        if os.path.exists(index_path):
            os.remove(index_path)  # 删除索引文件
            print(f"索引文件已删除: {index_path}")
        else:
            print(f"索引文件不存在，无需删除: {index_path}")