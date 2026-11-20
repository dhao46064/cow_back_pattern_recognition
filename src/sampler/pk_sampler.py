import random
from collections import defaultdict

from torch.utils.data import Sampler


class PKSampler(Sampler):

    def __init__(
        self,
        labels,
        P=16,
        K=4,
        batch_size=None
    ):
        """
        labels:
            dataset中每个样本对应的类别label

        P:
            每个batch采样多少个类别

        K:
            每个类别采样多少张图片

        batch_size:
            =P*K
        """

        self.labels = labels

        self.P = P
        self.K = K


        if batch_size is None:
            self.batch_size = P*K
        else:
            self.batch_size = batch_size


        # label -> index列表

        self.label_to_indices = defaultdict(list)


        for idx, label in enumerate(labels):

            self.label_to_indices[label].append(idx)



        self.labels_set = list(
            self.label_to_indices.keys()
        )


        # batch数量

        self.num_batches = len(labels)//self.batch_size



    def __iter__(self):

        batch_indexes = []


        for _ in range(self.num_batches):

            # 随机选择P个牛

            selected_labels = random.sample(
                self.labels_set,
                self.P
            )


            batch=[]


            for label in selected_labels:


                indices = self.label_to_indices[label]


                # 如果图片数量不足K张

                if len(indices) >= self.K:

                    sampled = random.sample(
                        indices,
                        self.K
                    )

                else:

                    sampled = random.choices(
                        indices,
                        k=self.K
                    )


                batch.extend(sampled)


            batch_indexes.append(batch)



        # 打乱batch顺序

        random.shuffle(batch_indexes)


        for batch in batch_indexes:

            yield batch



    def __len__(self):

        return self.num_batches