package verificacao;

import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.FileReader;
import java.io.FileWriter;

import weka.classifiers.Classifier;
import weka.classifiers.lazy.IBk;
import weka.core.Instance;
import weka.core.Instances;

public class Classificacao {
	
public static void main(String[] args) throws Exception{
		
		BufferedReader ler = null;
		ler = new BufferedReader(new FileReader("/home/jose/treino.arff"));
		Instances treino = new Instances(ler);
		treino.setClassIndex(treino.numAttributes() -1);
		
		ler = new BufferedReader(new FileReader("/home/jose/teste.arff"));
		Instances teste = new Instances(ler);
		teste.setClassIndex(teste.numAttributes() -1);
		
		ler.close();
		
		IBk knn = new IBk(3);
		knn.buildClassifier(treino);
		
		Instances rotulo = new Instances(teste);
		
		for (int i = 0; i < teste.numInstances(); i++){
			double clss = knn.classifyInstance(teste.instance(i));
			rotulo.instance(i).setClassValue(clss);
		}
		
		BufferedWriter escreve = new BufferedWriter(
				new FileWriter("/home/jose/classificado_k_3.arff"));
		escreve.write(rotulo.toString());
	
		
	}
	
	

}
