from utils.weighted_cross_entropy import SegmentationLosses

if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.cross_entropy_with_weights(a, b).item())
