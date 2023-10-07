import torch
import dgl

graph_data = {
    ('drug', 'interact', 'drug'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
    ('drug', 'interact', 'gene'): (torch.tensor([0, 1]), torch.tensor([3, 4])),
    ('drug', 'treats', 'disease'): (torch.tensor([1]), torch.tensor([5]))
}
g = dgl.heterograph(graph_data)
g.nodes['drug'].data['x'] = torch.rand(g.num_nodes('drug'), 3)
cuda_g = g.to('cuda:0')
print(cuda_g.device)

