import chromadb
from chromadb.config import Settings

# 1. 连接已创建的Chroma数据库（路径和你的agent.py一致）
chroma_client = chromadb.Client(Settings(
    persist_directory="./chroma_db",  # 和agent.py中的chroma_path一致
    anonymized_telemetry=False
))

# 2. 检查是否存在目标集合（你的集合名是"memories"）
try:
    collection = chroma_client.get_collection(name="memories")
    print("✅ 向量数据库集合（memories）创建成功！")

    # 3. 查看集合基本信息
    print(f"📊 集合中存储的总数据量：{collection.count()}")  # 数量>0说明有数据

    # 4. （可选）查询前5条数据，验证内容是否正确
    if collection.count() > 0:
        results = collection.get(
            limit=5,  # 取前5条
            include=["documents", "metadatas", "ids"]  # 包含文本、元数据、ID
        )
        print("\n📝 数据库中存储的示例数据：")
        for idx, (doc, meta, id) in enumerate(zip(results["documents"], results["metadatas"], results["ids"])):
            print(f"\n--- 第{idx + 1}条数据 ---")
            print(f"ID: {id}")
            print(f"文本内容（前100字）: {doc[:100]}...")
            print(f"元数据（文件类型/用户ID）: {meta.get('file_type')}/{meta.get('user_id')}")

except Exception as e:
    print(f"❌ 向量数据库验证失败：{str(e)}")
    # 若提示"Collection not found"，说明集合未创建，可手动创建
    chroma_client.create_collection(name="memories")
    print("⚠️ 已自动创建memories集合，请重新上传文件测试")