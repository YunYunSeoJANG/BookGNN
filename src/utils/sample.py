import torch

def sample_negative_edges_nocheck(data, num_users, num_items, device = None):
  # note computationally inefficient to check that these are indeed negative edges
  users = data.edge_label_index[0, :]
  items = torch.randint(num_users, num_users + num_items - 1, size = data.edge_label_index[1, :].size())

  if users.get_device() != -1: # on gpu 
    items = items.to(device)

  neg_edge_index = torch.stack((users, items), dim = 0)
  neg_edge_label = torch.zeros(neg_edge_index.shape[1])

  if neg_edge_index.get_device() != -1: # on gpu 
    neg_edge_label = neg_edge_label.to(device)
    
  return neg_edge_index, neg_edge_label

def sample_negative_edges(data, num_users, num_items, device=None):
    positive_users, positive_items = data.edge_label_index

    for i in range(len(positive_users)):
      if (positive_users[i]>=num_users or positive_items[i]<num_users):
        temp = positive_users[i]
        positive_users[i] = positive_items[i]
        positive_items[i] = temp

    #print(positive_users)
    #print(positive_items)
    #print(positive_users.min(), positive_users.max())
    #print(positive_items.min(), positive_items.max())
    #print(num_users, num_items)


    # Create a mask tensor with the shape (num_users, num_items)
    mask = torch.zeros(num_users, num_items, device=device, dtype=torch.bool)
    
    mask[positive_users, positive_items-num_users] = True

    # Flatten the mask tensor and get the indices of the negative edges
    flat_mask = mask.flatten()
    negative_indices = torch.where(~flat_mask)[0]

    # Sample negative edges from the negative_indices tensor
    sampled_negative_indices = negative_indices[
        torch.randint(0, negative_indices.size(0), size=(positive_users.size(0),), device=device)
    ]

    # Convert the indices back to users and items tensors
    users = torch.floor_divide(sampled_negative_indices, num_items)
    items = torch.remainder(sampled_negative_indices, num_items)
    items = items + num_users

    neg_edge_index = torch.stack((users, items), dim=0)
    neg_edge_label = torch.zeros(neg_edge_index.shape[1], device=device)

    return neg_edge_index, neg_edge_label

def sample_hard_negative_edges(data, model, num_users, num_items, device=None, batch_size=500, frac_sample = 1):
    with torch.no_grad():
        embeddings = model.get_embedding(data.edge_index)
        users_embeddings = embeddings[:num_users].to(device)
        items_embeddings = embeddings[num_users:].to(device)

    positive_users, positive_items = data.edge_label_index
    num_edges = positive_users.size(0)

    for i in range(len(positive_users)):
      if (positive_users[i]>=num_users or positive_items[i]<num_users):
        temp = positive_users[i]
        positive_users[i] = positive_items[i]
        positive_items[i] = temp
    
    # Create a boolean mask for all the positive edges
    positive_mask = torch.zeros(num_users, num_items, device=device, dtype=torch.bool)
    positive_mask[positive_users, positive_items-num_users] = True

    neg_edges_list = []
    neg_edge_label_list = []

    for batch_start in range(0, num_edges, batch_size):
        batch_end = min(batch_start + batch_size, num_edges)

        batch_scores = torch.matmul(
            users_embeddings[positive_users[batch_start:batch_end]], items_embeddings.t()
        )

        # Set the scores of the positive edges to negative infinity
        batch_scores[positive_mask[positive_users[batch_start:batch_end]]] = -float("inf")

        # Select the top k highest scoring negative edges for each playlist in the current batch
        # do 0.99 to filter out all pos edges which will be at the end 
        _, top_indices = torch.topk(batch_scores, int(frac_sample * 0.99 * num_items), dim=1)
        selected_indices = torch.randint(0, int(frac_sample * 0.99 *num_items), size = (batch_end - batch_start, ))
        top_indices_selected = top_indices[torch.arange(batch_end - batch_start), selected_indices] + num_users

        # Create the negative edges tensor for the current batch
        neg_edges_batch = torch.stack(
            (positive_users[batch_start:batch_end], top_indices_selected), dim=0
        )
        neg_edge_label_batch = torch.zeros(neg_edges_batch.shape[1], device=device)

        neg_edges_list.append(neg_edges_batch)
        neg_edge_label_list.append(neg_edge_label_batch)

    # Concatenate the batch tensors
    neg_edges = torch.cat(neg_edges_list, dim=1)
    neg_edge_label = torch.cat(neg_edge_label_list)

    return neg_edges, neg_edge_label