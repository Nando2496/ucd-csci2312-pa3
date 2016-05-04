#include <limits>

#include "KMeans.h"
#include "Exceptions.h"


namespace Clustering {

    KMeans::KMeans(unsigned int dim, unsigned int k, std::string filename, unsigned int maxIter) {
        if (k == 0)
            throw ZeroClustersEx();

        std::ifstream file(filename);

        if (!file)
            throw DataFileOpenEx(filename);

        __maxIter = maxIter;
        __numNonempty = 1;
        __dimensionality = dim;
        __k = k;

        __clusters = new Cluster *[k];
        for (unsigned int i = 0; i < k; ++i) __clusters[i] = new Cluster(dim);

        file >> *(__clusters[0]);

        file.close();

        __initCentroids = new Point *[k];
        for (unsigned int i = 0; i < k; ++i) __initCentroids[i] = new Point(dim);

        __clusters[0]->pickCentroids(k, __initCentroids);
    }


    KMeans::~KMeans() {
        for (unsigned int i = 0; i < __k; ++i) {
            delete __initCentroids[i];
            delete __clusters[i];
        }
        delete [] __initCentroids;
        delete [] __clusters;
    }

    unsigned int KMeans::getMaxIter() {
        return __maxIter;
    }
    unsigned int KMeans::getNumIters() {
        return __numIter;
    }
    unsigned int KMeans::getNumNonemptyClusters() {
        return __numNonempty;
    }
    unsigned int KMeans::getNumMovesLastIter() {
        return __numMovesLastIter;
    }

    Cluster &KMeans::operator[](unsigned int u) {
        return *(__clusters[u]);
    }
    const Cluster &KMeans::operator[](unsigned int u) const {
        return *(__clusters[u]);
    }


    std::ostream &operator<<(std::ostream &os, const KMeans &kmeans) {
        for (unsigned int i = 0; i < kmeans.__k; ++i) {
            os << kmeans[i];
        }
    }

    void KMeans::run() {
        int moves = 100;
        int iterator = 0;
        int nonempty = 0;

        while (moves > 0 && iterator < __maxIter) {
            moves = 0;

            for (int i = 0; i < __k; ++i) {
                for (int j = 0; j < __clusters[i]->getSize(); ++j)
                {
                    Cluster &c = *(__clusters[i]);

                    Point current_point(__dimensionality);
                    current_point = c[j];
                    int smallest_dist_index = 0;
                    double smallest_dist = current_point.distanceTo(*__initCentroids[0]);

                    for (int e = 0; e < __k; e++) {
                        if (current_point.distanceTo(*__initCentroids[e]) < smallest_dist) {
                            smallest_dist = current_point.distanceTo(*__initCentroids[e]);
                            smallest_dist_index = e;
                        }
                    }
                    Cluster::Move change_clusters(current_point, *__clusters[i], *__clusters[smallest_dist_index]);
                    change_clusters.perform();

                    for (int c = 0; c < __k; ++c) {
                        __clusters[c]->centroid.compute();
                    }
                    if (*__clusters[i] != *__clusters[smallest_dist_index]) {
                        moves++;
                    }
                }
            }

            iterator++;
        }

        Point inf(__dimensionality);

        for (int i = 0; i < __dimensionality; ++i) {
            inf[i] = std::numeric_limits<double>::max();
        }

        for (int i = 0; i < __k; ++i) {
            if (__clusters[i]->centroid.get() != inf) {
                ++nonempty;
            }
        }
        __numIter = iterator;
        __numMovesLastIter = moves;
        __numNonempty = nonempty;
    }

}

