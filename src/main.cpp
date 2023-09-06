#include <iostream>
#include <algorithm>
#include <chrono>
#include <torch/torch.h>

#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#include <stb_image_write.h>

using namespace torch;

// ------------------------------------------------------

class Image {
public:

    Image(const char* filepath) {
        unsigned char* rawdata = stbi_load(filepath, &width, &height, &bpp, 0);
        data = torch::zeros({ 3, height, width });
        for (int b = 0; b < 3; b++) {
            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {
                    data[b][y][x] = rawdata[width * bpp * (height - 1 - y) + x * bpp + b] / 255.0f;
                }
            }
        }
        stbi_image_free(rawdata);
    }

    Image(int width, int height) {
        data = torch::rand({3, height, width});
    }
    Image(const auto& data) :
        data(data),
        width(data.size(2)),
        height(data.size(1))
    {}

    Tensor data;
    int width;
    int height;
    int bpp;
};

class Intersection {
public:
    Tensor intersected;
    Tensor t;
    Tensor p;
    Tensor n;
    Tensor triangle;
    Tensor b;

    Intersection(int len) :
        intersected(torch::zeros({ len }).to(torch::kBool)),
        t(tensor({ 99999.0f }).unsqueeze(0).repeat({ len, 1 })),
        p(torch::zeros({ len, 3 })),
        n(torch::zeros({ len, 3 })),
        triangle(torch::zeros({ len }).to(torch::kInt32)),
        b(torch::zeros({ len, 3 }))
    {
    }
};

class Triangle {
public:

    const Tensor* positions;
    const Tensor* normals;
    Tensor indices;

    const Tensor* displaceParam;

    Tensor position(int i) const
    {
        return (*positions)[indices[i].item<int>()] + cat({ (*displaceParam)[indices[i].item<int>() * 2].expand(1), (*displaceParam)[indices[i].item<int>() * 2 + 1].expand(1), torch::zeros({1}) });
    }
    Tensor normal(int i) const
    {
        return normalized(position(i));
    }

    Tensor displacement(int i) const {
        return cat({ (*displaceParam)[indices[i].item<int>() * 2].expand(1), (*displaceParam)[indices[i].item<int>() * 2 + 1].expand(1), torch::zeros({1}) });
    }
};

class Ray {
public:
    Tensor orig;
    Tensor dir;
    Tensor weight;
};


class Scene {
public:
    void addTriangle(Triangle* triangle){
        triangles.push_back(triangle);
    }

    Intersection intersect(const Ray& ray) {
        int len = ray.orig.size(0);
        Intersection ints(len);
        int i = 0;
        for (auto& tri : triangles)
        {
            auto tmp = ::intersect(ray, *tri);
            
            for (int k = 0; k < ints.intersected.size(0); k++) {
                if (tmp.intersected[k].item<bool>() && 
                    (!ints.intersected[k].item<bool>() || tmp.t[k][0].item<float>() < ints.t[k][0].item<float>())
                    ) {
                    torch::NoGradGuard noGrad;
                    ints.intersected[k] = tmp.intersected[k];
                    ints.p[k] = tmp.p[k];
                    ints.n[k] = tmp.n[k];
                    ints.t[k] = tmp.t[k];
                    ints.triangle[k] = i;
                    ints.b[k] = tmp.b[k];
                }
            }

            ++i;
        }
        return ints;
    }

    std::vector<Triangle*> triangles;
};

class Camera {
public:

    Tensor p;
    Tensor dir;
    float orthoWidth;

    float orthoHeight;
    Tensor up;
    Tensor right;
    Tensor kernelCenter;
    Tensor offset;
    Tensor sigma;

    Ray getRay(int imageWidth, int imageHeight){
        orthoHeight = orthoWidth * imageHeight / imageWidth;
        int len = imageWidth * imageHeight;
        Ray ray;
        ray.orig = p.unsqueeze(0).repeat({ len, 1 });

        up = tensor({0, 1, 0}, TensorOptions().dtype(kFloat));
        right = cross(up, dir);

        Tensor u = torch::linspace(-0.5f, 0.5f, imageWidth) * orthoWidth;
        Tensor v = torch::linspace(-0.5f, 0.5f, imageHeight) * orthoHeight;

        kernelCenter = cartesian_prod({ v, u });

        sigma = tensor({ 1.0f / imageHeight, 1.0f / imageWidth });
        offset = cat({ torch::normal(0.0f, sigma[0].item<float>(), {len}).unsqueeze(1), torch::normal(0.0f, sigma[1].item<float>(), {len}).unsqueeze(1)}, 1);

        ray.orig += right * index_select(kernelCenter + offset, 1, tensor({ 1 }))
            + up * index_select(kernelCenter + offset, 1, tensor({ 0 }));

        ray.dir = dir.unsqueeze(0).repeat({ len, 1 });

        ray.weight = torch::ones({ len });
        
        return ray;
    }

    Tensor calcWeight(const Tensor& offset) {
        int len = offset.size(0);
        auto weight = torch::ones({ len });
        weight *= ((1 / sqrt(2.0f * M_PI * pow(sigma, 2))).unsqueeze(0) * torch::exp(-pow(offset, 2) / (2 * pow(sigma, 2)).unsqueeze(0))).prod(1);
        return weight;
    }

    Tensor calcWeight() {
        return calcWeight(offset);
    }

    Tensor positionToScreenSpace(const Tensor& pos) {
        Tensor x = dotBatch(pos - p, right) / orthoWidth;
        Tensor y = dotBatch(pos - p, up) / orthoHeight;
        return cat({y, x}, 1);
    }
};

// ------------------------------------------------------

Tensor dotBatch(const Tensor& a, const Tensor& b) {
    return (a * b).sum(1, true);
}

Tensor crossBatch(const Tensor& a, const Tensor& b) {
    return cross(a, b, 1);
}

Tensor normalized(const Tensor& x) {
    return div(x, torch::norm(x, nullopt, 0, true));
}
Tensor normalizedBatch(const Tensor& x) {
    return div(x, torch::norm(x, nullopt, 1, true));
}

Tensor safeValue(const Tensor& x) {
    return torch::where(isnan(x), tensor(0), x);
}

Tensor select(const Tensor& mask, const Tensor& a, const Tensor& b) {
    return mask.to(kFloat) * safeValue(a) + (1 - mask.to(kFloat)) * safeValue(b);
}

Intersection intersect(const Ray& ray, const Triangle& tri) {

    int len = ray.orig.size(0);
    Intersection ints(len);

    const auto& p0 = tri.position(0).unsqueeze(0);
    const auto& p1 = tri.position(1).unsqueeze(0);
    const auto& p2 = tri.position(2).unsqueeze(0);

    auto ng = normalizedBatch(crossBatch(p1 - p0, p2 - p0));
    ints.t = (dotBatch(p0 - ray.orig, ng) / dotBatch(ray.dir, ng));
    ints.p = ray.orig + ints.t * ray.dir;

    auto v0 = p1 - p0;
    auto v1 = p2 - p0;
    auto v2 = ints.p - p0;
    auto d00 = dotBatch(v0, v0);
    auto d01 = dotBatch(v0, v1);
    auto d11 = dotBatch(v1, v1);
    auto d20 = dotBatch(v2, v0);
    auto d21 = dotBatch(v2, v1);
    auto div = (d00 * d11 - d01 * d01);
    auto b1 = (d11 *  d20 - d01 * d21) / div;
    auto b2 = (d00 *  d21 - d01 * d20) / div;
    auto b0 = 1 - b1 - b2;

    ints.intersected = ((0 <= b0 & b0 <= 1) & (0 <= b1 & b1 <= 1) & (0 <= b2 & b2 <= 1)).squeeze(1);

    const auto& n0 = tri.normal(0).unsqueeze(0);
    const auto& n1 = tri.normal(1).unsqueeze(0);
    const auto& n2 = tri.normal(2).unsqueeze(0);
    ints.n = ng.repeat({len, 1});

    ints.b = cat({b0, b1, b2}, 1);

    return ints;
}

bool importAssimp(
    std::vector<int>* indices,
    std::vector<float>* positions,
    std::vector<float>* normals,
    const std::string& filepath)
{
    Assimp::Importer importer;
    const aiScene* scene = importer.ReadFile(filepath, aiProcess_Triangulate | aiProcess_JoinIdenticalVertices);
    if (!scene) { return false; }
    if (!scene->HasMeshes()) { return fakse; }

    aiMesh* mesh = scene->mMeshes[0];

    int triNum = mesh->mNumFaces;
    indices->reserve(triNum * 3);
    for (int i = 0; i < triNum; ++i) {
        const aiFace& face = mesh->mFaces[i];
        if (face.mNumIndices == 3) {
            indices->push_back(face.mIndices[0]);
            indices->push_back(face.mIndices[1]);
            indices->push_back(face.mIndices[2]);
        }
    }

    int vertNum = mesh0->mNumVertices;
    positions->reserve(vertNum * 3);
    normals->reserve(vertNum * 3);
    for (int i = 0; i < vertNum; ++i) {
        const aiVector3D& v = mesh->mVertices[i];
        const aiVector3D& n = mesh->mNormals[i];
        positions->push_back(v.x);
        positions->push_back(v.y);
        positions->push_back(v.z);
        normals->push_back(normal.x);
        normals->push_back(normal.y);
        normals->push_back(normal.z);
    }

    return true;
}

// ------------------------------------------------------

int main() {

    // -------- 色々初期化

    int width = 128;
    int height = width;
    int bpp = 3;
    unsigned char* pixels = (unsigned char*)malloc(width * height * bpp);

    auto start_time = std::chrono::high_resolution_clock::now();

    // いったん vector に読んで Tensor にコピー
    // 無駄に二重に存在してるけどまあ見逃す
    std::vector<int> assimpIndices;
    std::vector<float> assimpPositions;
    std::vector<float> assimpNormals;
    if (!importAssimp(&assimpIndices, &assimpPositions, &assimpNormals, "data/gasshuku2.obj"))
    {
        printf("damedesu\n");
        return 1;
    }

    auto positions = torch::full({ (int)assimpPositions.size(), 3}, 0.0f, TensorOptions().dtype(kFloat)) * 0.1f;
    auto normals = torch::full({ (int)assimpNormals.size(), 3 }, 0.0f, TensorOptions().dtype(kFloat)) * 0.1f;
    for (unsigned int i = 0; i < assimpPositions.size() / 3; i++) {
        positions[i][0] = assimpPositions[i * 3 + 0];
        positions[i][1] = assimpPositions[i * 3 + 1];
        positions[i][2] = assimpPositions[i * 3 + 2];
    }
    for (unsigned int i = 0; i < assimpNormals.size() / 3; i++) {
        normals[i][0] = assimpNormals[i * 3 + 0];
        normals[i][1] = assimpNormals[i * 3 + 1];
        normals[i][2] = assimpNormals[i * 3 + 2];
    }

    auto inputTensor = torch::zeros({ positions.size(0) * 2}, requires_grad());

    std::vector<Triangle> triangles(assimpIndices.size() / 3);
    for (unsigned int i = 0; i < triangles.size(); i++) {
        auto& tri = triangles[i];
        tri.positions = &positions;
        tri.normals = &normals;
        tri.indices = tensor({ assimpIndices[i * 3 + 0], assimpIndices[i * 3 + 1], assimpIndices[i * 3 + 2] }, torch::kInt32);
        tri.displaceParam = &inputTensor;
    }

    optim::Adam optimizer({ inputTensor }, optim::AdamOptions(0.01f));

    // -------- 描画本体
    auto render = [&](const Tensor& inputTensor) {

        Scene scene;
        for (unsigned int i = 0; i < triangles.size(); i++) {
            auto& tri = triangles[i];
            scene.addTriangle(&tri);
        }

        Camera camera;
        camera.p = tensor({ 0.0f, 0.0f, -2.0f }, TensorOptions().dtype(kFloat));
        camera.dir = tensor({ 0, 0, 1 }, TensorOptions().dtype(kFloat));
        camera.orthoWidth = 2.0f;

        Ray ray = camera.getRay(width, height);

        auto isct = scene.intersect(ray);

        // ポリゴンが移動した際に交点も移動するとして、weight 計算
        auto p = torch::zeros({ isct.triangle.size(0), 3 });
        for (int i = 0; i < isct.triangle.size(0); i++) {
            if (!isct.intersected[i].item<bool>()) {
                continue;
            }
            auto p0 = triangles[isct.triangle[i].item<int>()].position(0);
            auto p1 = triangles[isct.triangle[i].item<int>()].position(1);
            auto p2 = triangles[isct.triangle[i].item<int>()].position(2);
            p[i] = isct.b[i][0] * p0 + isct.b[i][1] * p1 + isct.b[i][2] * p2;
        }
        auto disp1 = camera.positionToScreenSpace(p);
        auto disp0 = disp1.detach();
        auto w1 = camera.calcWeight(camera.offset + (disp1 - disp0));
        auto w0 = camera.calcWeight();
        ray.weight *= w1 / w0;

        // ポリゴンの色は結局定数に
        auto radiance = select(isct.intersected.unsqueeze(1).repeat({ 1, 3 }), torch::full({ height * width, 3 }, 0.9f), torch::full({ height * width, 3 }, 0.1f));

        // 近傍の radiance を適当に参照して control variable 計算
        auto neighbourRadiance = radiance.slice(0, width);
        neighbourRadiance = cat({ neighbourRadiance, neighbourRadiance[neighbourRadiance.size(0) - 1].unsqueeze(0).repeat({width, 1}) });
        radiance *= ray.weight.unsqueeze(1).repeat({ 1, 3 });
        auto alpha = -neighbourRadiance;
        radiance += alpha * (w1/w0 - 1).unsqueeze(1).repeat({ 1, 3 });

        return radiance.index_select(1, tensor({ 0 })).flatten(); // Rだけ
    };

    // -------- 最適化と出力

    auto target = Image("data/target.png").data;
    target = target.index_select(0, tensor({ 0 })); // Rだけ

    int i = 0;
    while(true) {

        // 最適化
        torch::Tensor prediction = torch::zeros({ width * height * 1 });
        int samples = 8;
        for (int sample = 0; sample < samples; sample++) {
            prediction += render(inputTensor);
        }
        prediction /= samples;

        torch::Tensor loss = (prediction.flatten() - target.flatten()).pow(2).sum();

        loss.backward();
        optimizer.step();
        optimizer.zero_grad();

        printf("%d %f\n", i, loss.item<float>());

        // 結果出力
        if (i % 1 == 0) {
            Tensor result = prediction.repeat({3});

            for (int b = 0; b < 3; b++) {
                for (int y = 0; y < height; y++) {
                    for (int x = 0; x < width; x++) {
                        pixels[width * bpp * (height - 1 - y) + x * bpp + b] = std::clamp((int)std::roundf(result[width * height * b + width * y + x].item<float>() * 255), 0, 255);
                        //pixels[width * bpp * (height - 1 - y) + x * bpp + b] = std::clamp((int)std::roundf(diff[width * height * b + width * y + x].item<float>() * 255), 0, 255);
                    }
                }
            }
            auto filename = std::to_string(i);
            while (filename.length() < 3) {
                filename = "0" + filename;
            }
            filename += ".png";
            stbi_write_png(filename.c_str(), width, height, bpp, pixels, width * bpp);
        }

        // 時間を見て打ち切る
        auto current_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(current_time - start_time).count() / 1000;
        printf("%d\n", duration);
        if (duration > (300 - 10) * 1000) {
            break;
        }

        ++i;
    }

    // アニメーションさせたときのために、最後の状態の画像を追加で 10 枚出しとく
    for (int k = 0; k < 10; k++) {
        ++i;
        auto filename = std::to_string(i);
        while (filename.length() < 3) {
            filename = "0" + filename;
        }
        filename += ".png";
        stbi_write_png(filename.c_str(), width, height, bpp, pixels, width * bpp);
    }

    stbi_image_free(pixels);

    return 0;
}