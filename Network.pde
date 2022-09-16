
abstract class Network{
  float loss;
  Network cNet;
  Network sNet;
  Layer[] layers;
  FixedImageMap[] image;
  abstract void forward(ImageMap[] x);
  abstract void forward();
  abstract void backward();
}

class VGG19 extends Network{
  
  VGG19(){
    layers = new Layer[37];
    layers[0] = new ConvLayer(3, 64, "vgg19/conv1_1.txt");
    layers[1] = new ReLU();                                // add style loss_1 gradient here, w=0.2
    layers[2] = new ConvLayer(64, 64, "vgg19/conv1_2.txt");      
    layers[3] = new ReLU();
    layers[4] = new Pooling();                             // size : 512 -> 256
    println("initialized 1");
    layers[5] = new ConvLayer(64, 128, "vgg19/conv2_1.txt");
    layers[6] = new ReLU();                                // add style loss_2 gradient here, w=0.2
    layers[7] = new ConvLayer(128, 128, "vgg19/conv2_2.txt");    
    layers[8] = new ReLU();
    layers[9] = new Pooling();                             // size : 256 -> 128
    println("initialized 2");
    layers[10] = new ConvLayer(128, 256, "vgg19/conv3_1.txt");
    layers[11] = new ReLU();                               // add style loss_3 gradient here, w=0.2
    layers[12] = new ConvLayer(256, 256, "vgg19/conv3_2.txt");    
    layers[13] = new ReLU();
    layers[14] = new ConvLayer(256, 256, "vgg19/conv3_3.txt");
    layers[15] = new ReLU();
    layers[16] = new ConvLayer(256, 256, "vgg19/conv3_4.txt");
    layers[17] = new ReLU();
    layers[18] = new Pooling();                             // size : 128 -> 64
    println("initialized 3");
    layers[19] = new ConvLayer(256, 512, "vgg19/conv4_1.txt");
    layers[20] = new ReLU();                                // add style loss_4 gradient here, w=0.2
    layers[21] = new ConvLayer(512, 512, "vgg19/conv4_2.txt");    
    layers[22] = new ReLU();                                // add content loss gradient here
    layers[23] = new ConvLayer(512, 512, "vgg19/conv4_3.txt");    
    layers[24] = new ReLU();
    layers[25] = new ConvLayer(512, 512, "vgg19/conv4_4.txt");
    layers[26] = new ReLU();
    layers[27] = new Pooling();                             // size : 64 -> 32
    println("initialized 4");
    layers[28] = new ConvLayer(512, 512, "vgg19/conv5_1.txt");
    layers[29] = new ReLU();                                // add style loss_5 gradient here, w=0.2
    layers[30] = new ConvLayer(512, 512, "vgg19/conv5_2.txt");    
    layers[31] = new ReLU();
    layers[32] = new ConvLayer(512, 512, "vgg19/conv5_3.txt");
    layers[33] = new ReLU();
    layers[34] = new ConvLayer(512, 512, "vgg19/conv5_4.txt");
    layers[35] = new ReLU();
    layers[36] = new Pooling();
    println("initialized 5");
  }
  
  VGG19(Network C, Network S){
    this();
    cNet = C;
    sNet = S;
  }
  
  void forward(){
    println("start");
    ImageMap[] y = layers[0].forward(image);
    for(int i=1; i<34; i++){ // change 10 to 34
      println("layer "+i+" start");
      y = layers[i].forward(y);
    } 
    println("finished");
  }
  void forward(ImageMap[] X){
    image = new FixedImageMap[3];
    for(int i=0; i<3; i++){
      image[i] = new FixedImageMap(X[i].H, X[i].W);
      image[i].copy(X[i]);
    }
    forward();
  }
  
  void backward(){ // change 7 to 31
    loss = 0;
    ImageMap[] dLdy = new ImageMap[layers[32].x.length];
    for(int i=0; i<layers[32].x.length; i++) 
      dLdy[i] = new ImageMap(layers[32].x[0].H, layers[32].x[0].W);
      
    println("backward");
    ImageMap[] dLdx = layers[31].backward(dLdy);
    for(int i=30; i>=0; i--){
      println("layer "+i+" backward");
      dLdx = layers[i].backward(dLdx);
      if(LsIndex.contains(i)){
        ImageMap[] Grad = getLsGrad(i, LsWeight.get(LsIndex.indexOf(i)));
        for(int j=0; j<dLdx.length; j++)
          dLdx[j].add(Grad[j]);
      }
      if(LcIndex==i){
        ImageMap[] Grad = getLcGrad(i);
        for(int j=0; j<dLdx.length; j++)
          dLdx[j].add(Grad[j]);
      }
    }
    
    for(int i=0; i<3; i++)
      image[i].optimize(dLdx[i]);
    println(loss);
  }
  
  ImageMap[] getLcGrad(int l){
    ImageMap[] x = layers[l].x;
    ImageMap[] c = cNet.layers[l].x;
    int N = x.length;
    ImageMap[] dLdx = new ImageMap[N];
    for(int I=0; I<N; I++){
      dLdx[I] = new ImageMap(x[I].H, x[I].W);
      dLdx[I].add(x[I]);
      dLdx[I].sub(c[I]);
      dLdx[I].mult(alpha);
      for(int i=0; i<x[I].H; i++)
        for(int j=0; j<x[I].W; j++){
          loss += 0.5*dLdx[I].get(i,j)*dLdx[I].get(i,j)/alpha;
          //print(loss+" ,");
        }
    }
    return dLdx;
  }
  
  ImageMap[] getLsGrad(int l, float w){
    ImageMap[] x = layers[l].x;
    ImageMap[] s = sNet.layers[l].x;
    int N = x.length;
    ImageMap[] dLdx = new ImageMap[N];
    float[][] Grams = new float[N][N];
    float[][] Gramx = new float[N][N];
    for(int I=0; I<N; I++)
      for(int J=0; J<N; J++){
        Gramx[I][J] = x[I].product(x[J]);
        Grams[I][J] = s[I].product(s[J]);
      }
      
    for(int I=0; I<N; I++)
      for(int J=0; J<N; J++){
        float temp = (float)(Gramx[I][J]-Grams[I][J])/(2*N*(x[I].H*x[I].W));
        loss += beta*w*temp*temp;
        // println(loss);
      }
      
    for(int I=0; I<N; I++){
      dLdx[I] = new ImageMap(x[I].H, x[I].W);
      for(int i=0; i<x[I].H; i++)
        for(int j=0; j<x[I].W; j++){
          float sum=0;
          for(int J=0; J<N; J++){
            float temp = (Gramx[I][J]-Grams[I][J])/(N*x[I].H*x[I].W);
            temp *= x[J].get(i,j)/(N*x[I].H*x[I].W);
            sum += temp;
          }
          dLdx[I].set(i,j,sum);
          //println("this is sum : "+sum);
        }
      dLdx[I].mult(beta*w);
    }    
    return dLdx;
  }
  
  
}
