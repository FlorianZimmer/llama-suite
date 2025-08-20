# pip install gguf
import gguf, sys
m = gguf.GGUFReader(sys.argv[1])
md = {k:v[0] for k,v in m.get_kv_data().items()}
print("n_head =", md.get("attention.head_count"))
print("n_head_kv =", md.get("attention.head_count_kv"))