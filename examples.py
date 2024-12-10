import torch
import numpy as np
 
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import pointnet2_utils
import matplotlib.pyplot as plt
 
def test_furthest_point_sampling():
    b = 10
    c = 2
    N = 1000
    n = 10
    pool = torch.randn(b, N, c, requires_grad=True).float().cuda()
    indices = pointnet2_utils.furthest_point_sample(pool, n).long()
    indices = indices.detach().cpu() # [b,n]
    pool = pool.detach().cpu() # [b,c,m]
    sampled = torch.gather(pool, 1, indices.unsqueeze(-1).expand(b, n, c))
    sampled = sampled.numpy()
    pool = pool.numpy()
    print(indices)
    plt.figure()
    plt.subplot(121)
    plt.plot(pool[0,:,0], pool[0,:,1], 'b.')
    plt.plot(sampled[0,:,0], sampled[0,:,1], 'r.')
    plt.subplot(122)
    plt.plot(pool[1,:,0], pool[1,:,1], 'b.')
    plt.plot(sampled[1,:,0], sampled[1,:,1], 'r.')
    plt.savefig('furthest_point_sample.png')


def test_ball_query():
    b = 3  # batch size
    c = 2  # dimension of coordinates (x, y)
    num_db = 1000  # number of points in the database
    num_query = 1  # number of query points
    num_neighbors = 50  # number of neighbors to search for each query
    radius = 0.5  # search neighbor radius for each query

    # Create random coordinates for db and query points
    db_xyz = torch.randn(b, num_db, c, requires_grad=True).float().cuda() 
    query_xyz = torch.randn(b, num_query, c, requires_grad=True).float().cuda()

    # Perform ball query to get the neighbor indices
    output_ids = pointnet2_utils.ball_query(radius, num_neighbors, db_xyz, query_xyz)  # [b, num_query, num_neighbors]
    output_ids = output_ids.long()
    # Use the output_ids to get the corresponding coordinates from db_xyz
    # output_ids has shape [b, num_query, num_neighbors]
    # output_xyz should have the shape [b, num_query, num_neighbors, 2]
    
    output_ids = output_ids.view(b, num_query*num_neighbors)  # [b, num_query*num_neighbors]
    output_xyz = torch.gather(db_xyz, 1, output_ids.unsqueeze(-1).repeat(1, 1, c)) # [b, num_query*num_neighbors, c]
    output_xyz = output_xyz.view(b, num_query, num_neighbors, c)

    # Convert to numpy for visualization
    output_xyz = output_xyz.detach().cpu().numpy()
    db_xyz = db_xyz.detach().cpu().numpy()
    query_xyz = query_xyz.detach().cpu().numpy()

    # Plot the results
    c
    plt.figure(figsize=[10,10])
    plt.subplot(111)
    plt.plot(db_xyz[0, :, 0], db_xyz[0, :, 1], 'b.', markersize=2)
    plt.plot(query_xyz[0, 0, 0], query_xyz[0, 0, 1], 'r.', markersize=10)
    plt.plot(output_xyz[0, 0, :, 0], output_xyz[0, 0, :, 1], 'g.', markersize=10)
    # for i in range(num_neighbors):
    #     plt.plot(output_xyz[0, :, i, 0], output_xyz[0, :, i, 1], 'g.')
    # plt.savefig('ball_query.png')
    plt.show()


    return output_xyz

 
if __name__=='__main__':
    # test_furthest_point_sampling()
    test_ball_query()
