import pathway as pw

test1 = pw.io.fs.read(path = r"dynamic_kg_updation\files-for-indexing", format = "plaintext", mode="streaming")
output1 = pw.io.fs.write(test1, filename=r"dynamic_kg_updation\out1.json", format="json")

pw.run()
