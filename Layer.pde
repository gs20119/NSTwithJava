

abstract class Layer{
  int IN, OUT;
  ImageMap[] x;
  abstract ImageMap[] forward(ImageMap[] X);
  abstract ImageMap[] backward(ImageMap[] dLdy);
}


class ConvLayer extends Layer{
  int fSize = 1;
  Filter[] filters;
  
  ConvLayer(int in, int out, String loc){ 
    IN = in; OUT = out;
    filters = new Filter[OUT];
    String[] paramstr = loadStrings(loc);
    ArrayList<Float> params = new ArrayList();
    for(String line : paramstr){
      String[] splits = line.split("\\[|\\s|,|\\]");
      for(String num : splits){
        if(num.length()==0) continue;
        float n = Float.valueOf(num);
        params.add(n);
      }
    }
    int nFilm = (2*fSize+1)*(2*fSize+1);
    int nFilter = IN*nFilm;
    for(int i=0; i<OUT; i++){
      ArrayList<Float> fparams = new ArrayList();
      for(int j=0; j<nFilter; j++) fparams.add(params.get(i*nFilter+j));
      float bias = params.get(OUT*nFilter+i);
      filters[i] = new Filter(IN, fSize, fparams, bias);
    }
  }
  
  ImageMap[] forward(ImageMap[] X){
    x = X;
    ImageMap[] y = new ImageMap[OUT];
    for(int i=0; i<OUT; i++)
      y[i] = filters[i].Conv(x);
    println("Conv forward " + y[2].get(4,4) + ".");
    return y;
  }
  
  ImageMap[] backward(ImageMap[] dLdy){
    println(IN+" "+OUT+" "+dLdy.length);
    ImageMap[] dLdx = new ImageMap[IN];
    for(int j=0; j<IN; j++) 
      dLdx[j] = new ImageMap(x[j].H, x[j].W);
    for(int i=0; i<OUT; i++){
      ImageMap[] dLdx_ = filters[i].backConv(dLdy[i]);
      for(int j=0; j<IN; j++)
        dLdx[j].add(dLdx_[j]);
    }print("Conv backward " + dLdx[2].get(4,4) + ".");
    return dLdx;
  }
}


class Pooling extends Layer{
  int size=2;
  Pooling(){}
  ImageMap[] forward(ImageMap[] X){ 
    x = X; OUT = X.length;
    ImageMap[] y = new ImageMap[OUT];
    for(int i=0; i<OUT; i++){
      y[i] = new ImageMap(x[i].H/size, x[i].W/size);
      for(int I=0; I<y[i].H; I++)
        for(int J=0; J<y[i].W; J++){
          float sum=0;
          for(int a=0; a<size; a++)
            for(int b=0; b<size; b++)
              sum += x[i].get(a+I*size,b+J*size);
          y[i].set(I, J, sum/(size*size));
        }
    }
    return y;
  }
  ImageMap[] backward(ImageMap[] dLdy){ 
    ImageMap[] dLdx = new ImageMap[OUT];
    for(int i=0; i<OUT; i++){
      dLdx[i] = new ImageMap(x[i].H, x[i].W);
      for(int I=0; I<dLdy[i].H; I++)
        for(int J=0; J<dLdy[i].W; J++){
          float g = (float)dLdy[i].get(I,J)/(size*size);
          for(int a=0; a<size; a++)
            for(int b=0; b<size; b++)
              dLdx[i].set(I*size+a, J*size+b, g);
        }
    }print("Pooling backward " + dLdx[2].get(2,2) + ".");
    return dLdx; 
  }
}



class ReLU extends Layer{
  ReLU(){ }
  ImageMap[] forward(ImageMap[] X){ 
    x = X; OUT = X.length;
    ImageMap[] y = new ImageMap[OUT];
    for(int i=0; i<OUT; i++){
      y[i] = new ImageMap(x[i].H, x[i].W);
      for(int I=0; I<y[i].H; I++)
        for(int J=0; J<y[i].W; J++)
          if(x[i].get(I,J)>0) y[i].set(I,J,x[i].get(I,J));
    }
    return y;
  }
  ImageMap[] backward(ImageMap[] dLdy){ 
    ImageMap[] dLdx = new ImageMap[OUT];
    for(int i=0; i<OUT; i++){
      dLdx[i] = new ImageMap(x[i].H, x[i].W);
      for(int I=0; I<dLdx[i].H; I++)
        for(int J=0; J<dLdx[i].W; J++){
          if(x[i].get(I,J)>0) dLdx[i].set(I,J,dLdy[i].get(I,J));
          else dLdx[i].set(I,J,0);
          //if(I==2 && J==2) print("ReLU backward " + dLdx[i].get(I,J) + ".");
        }
    }
    return dLdx; 
  }
}
