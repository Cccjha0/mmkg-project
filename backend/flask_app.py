import pickle
from flask import Flask, request
from flask_cors import CORS, cross_origin
import json
import os
import pathlib
from flask import send_from_directory

# 👇 这里修复！导入你的 Search 类
from app.services.search import Search

MAX_NODE_CONNECTIONS = Search.MAX_GRAPH_LINKS

def create_app():
    app = Flask(__name__)
    CORS(app)

    @app.route("/health")
    def health():
        return {"status": "ok"}

    # 图片接口
    @app.route('/images/<entity_id>/<filename>')
    def serve_image(entity_id, filename):
        image_root = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "../data/datasets/openbg_img/raw/OpenBG-IMG_images"
            )
        )
        folder = os.path.join(image_root, entity_id)
        return send_from_directory(folder, filename)

    # 搜索接口
    @app.route("/search/<k>/<n>/<p>/<query>")
    @cross_origin()
    def get(k, n, p, query):
        lang = request.args.get('lang', 'en')

        base_dir = pathlib.Path(__file__).parent.parent
        data_dir = base_dir / "data" / "datasets" / "openbg_img" / "processed"

        search_path = data_dir / "search.pkl"
        csv_path = data_dir / "data.csv"
        metadata_path = data_dir / "metadata.json"

        if os.path.exists(search_path):
            with open(search_path, "rb") as f:
                search = pickle.load(f)
        else:
            search = Search(file=str(csv_path))
            search.save(str(search_path))

        if os.path.exists(metadata_path):
            search.load_metadata(str(metadata_path))

        result = search(
            query=query,
            k=int(k),
            n=int(n),
            p=int(p),
            lang=lang
        )

        return json.dumps(result, ensure_ascii=False)

    # 获取节点完整连接信息接口
    @app.route("/node_connections/<node_id>")
    @cross_origin()
    def get_node_connections(node_id):
        lang = request.args.get('lang', 'en')

        print(f"\n=== 获取节点连接信息 ===")
        print(f"节点ID: {node_id}")
        print(f"语言: {lang}")

        base_dir = pathlib.Path(__file__).parent.parent
        data_dir = base_dir / "data" / "datasets" / "openbg_img" / "processed"

        search_path = data_dir / "search.pkl"
        csv_path = data_dir / "data.csv"
        metadata_path = data_dir / "metadata.json"

        # 加载 search 对象
        if os.path.exists(search_path):
            with open(search_path, "rb") as f:
                search = pickle.load(f)
        else:
            search = Search(file=str(csv_path))
            search.save(str(search_path))

        if os.path.exists(metadata_path):
            search.load_metadata(str(metadata_path))

        # 获取该节点的所有连接
        connections = []

        # 检查 triples 中是否有这个节点
        print(f"节点在 triples 中: {node_id in search.triples}")

        # 从 triples 中查找与该节点相关的所有连接（正向：该节点指向其他节点）
        neighbours = search.triples.get(node_id, tuple([]))

        print(f"正向邻居数量: {len(neighbours)}")
        if len(neighbours) > 0:
            print(f"正向邻居示例: {list(neighbours)[:5]}")

        for neighbour_id in neighbours:
            # 获取关系
            relation_key = f"{node_id}_{neighbour_id}"
            relations = search.relations.get(relation_key, [])

            print(f"  正向邻居: {neighbour_id}, 关系: {relations}")

            # 获取邻居节点的元数据
            neighbour_meta = search.metadata.get(neighbour_id, {})

            # 根据语言获取标签
            if lang == "zh":
                neighbour_label = neighbour_meta.get("label_zh", neighbour_meta.get("label", neighbour_id))
            else:
                neighbour_label = neighbour_meta.get("label_en", neighbour_meta.get("label", neighbour_id))

            # 为每个关系创建连接记录
            for relation_id in relations:
                relation_label = search.get_relation_label(relation_id, lang)
                relation_label_zh = search.get_relation_label(relation_id, "zh")
                relation_label_en = search.get_relation_label(relation_id, "en")

                connections.append({
                    "node": {
                        "id": neighbour_id,
                        "label": neighbour_label,
                        "label_zh": neighbour_meta.get("label_zh", ""),
                        "label_en": neighbour_meta.get("label_en", ""),
                        "metadata": neighbour_meta
                    },
                    "relation": {
                        "id": relation_id,
                        "relation": relation_label,
                        "relation_zh": relation_label_zh,
                        "relation_en": relation_label_en
                    }
                })
                if len(connections) >= MAX_NODE_CONNECTIONS:
                    break
            if len(connections) >= MAX_NODE_CONNECTIONS:
                break

        # 同时检查反向连接（其他节点指向该节点）
        # 遍历所有节点，查找哪些节点连接到当前节点
        print(f"\n开始检查反向连接...")
        reverse_count = 0
        source_items = search.triples.items() if len(connections) < MAX_NODE_CONNECTIONS else []
        for source_node, targets in source_items:
            if node_id in targets:
                reverse_count += 1
                if reverse_count <= 3:
                    print(f"  反向连接: {source_node} -> {node_id}")

                # 获取关系
                relation_key = f"{source_node}_{node_id}"
                relations = search.relations.get(relation_key, [])

                # 获取源节点的元数据
                source_meta = search.metadata.get(source_node, {})

                if lang == "zh":
                    source_label = source_meta.get("label_zh", source_meta.get("label", source_node))
                else:
                    source_label = source_meta.get("label_en", source_meta.get("label", source_node))

                for relation_id in relations:
                    relation_label = search.get_relation_label(relation_id, lang)
                    relation_label_zh = search.get_relation_label(relation_id, "zh")
                    relation_label_en = search.get_relation_label(relation_id, "en")

                    connections.append({
                        "node": {
                            "id": source_node,
                            "label": source_label,
                            "label_zh": source_meta.get("label_zh", ""),
                            "label_en": source_meta.get("label_en", ""),
                            "metadata": source_meta
                        },
                        "relation": {
                            "id": relation_id,
                            "relation": relation_label,
                            "relation_zh": relation_label_zh,
                            "relation_en": relation_label_en
                        }
                    })
                    if len(connections) >= MAX_NODE_CONNECTIONS:
                        break
                if len(connections) >= MAX_NODE_CONNECTIONS:
                    break

        print(f"反向连接数量: {reverse_count}")
        print(f"最终总连接数量: {len(connections)}")
        print(f"========================\n")

        return json.dumps({
            "connections": connections,
            "total": len(connections)
        }, ensure_ascii=False)

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(host="127.0.0.1", port=5000, debug=True)
