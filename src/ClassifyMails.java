
import com.aliasi.classify.*;
import com.aliasi.lm.NGramProcessLM;
import com.aliasi.util.Files;

import java.io.*;


public class ClassifyMails {

	private static String[] CATEGORIES = {"SPAM", "NOSPAM"};

	private static int NGRAM_SIZE = 6;

	public static void main(String[] args)
			throws ClassNotFoundException, IOException {

		DynamicLMClassifier<NGramProcessLM> classifier;


		if (args[0].equals("learn")) {
			classifier = DynamicLMClassifier.createNGramProcess(CATEGORIES, NGRAM_SIZE);

			File[] trainingDir = new File[CATEGORIES.length];

			trainingDir[1] = new File(args[2]);
			trainingDir[0] = new File(args[1]);

			for (int i = 0; i < CATEGORIES.length; ++i) {
				File classDir = trainingDir[i];
				if (!classDir.isDirectory()) {
					String msg = "Could not find training directory="
							+ classDir
							+ "\nHave you unpacked 4 newsgroups?";
					System.out.println(msg); // in case exception gets lost in shell
					throw new IllegalArgumentException(msg);
				}

				String[] trainingFiles = classDir.list();
				for (int j = 0; j < trainingFiles.length; ++j) {
					File file = new File(classDir, trainingFiles[j]);
					String text = Files.readFromFile(file, "ISO-8859-1");
					System.out.println("Training on " + CATEGORIES[i] + "/" + trainingFiles[j]);
					Classification classification = new Classification(CATEGORIES[i]);
					Classified<CharSequence> classified = new Classified<CharSequence>(text, classification);
					classifier.handle(classified);
				}
			}
			System.out.println("Schreibe Model ");

			FileOutputStream fos = new FileOutputStream(new File("model"));
			ObjectOutputStream objectOutputStream = new ObjectOutputStream(fos);
			classifier.compileTo(objectOutputStream);

		} else if (args[0].equals("classify")) {
			//TESTING_DIR = ;

			File modelFile = new File("model");
			System.out.println("Lese Model " + modelFile);
			FileInputStream fileIn = new FileInputStream(modelFile);
			ObjectInputStream objIn = new ObjectInputStream(fileIn);


			//compiling
			System.out.println("Compiling");
			@SuppressWarnings("unchecked")
					JointClassifier<CharSequence> compiledClassifier
					= (JointClassifier<CharSequence>) objIn.readObject();
			objIn.close();

			BufferedWriter bw = new BufferedWriter(new FileWriter(new File(args[2])));

			File classDir = new File(args[1]);
			String[] testingFiles = classDir.list();
			for (int j = 0; j < testingFiles.length; ++j)
			{
				String text = Files.readFromFile(new File(classDir, testingFiles[j]), "ISO-8859-1");
				JointClassification jc = compiledClassifier.classify(text);
				String bestCategory = jc.bestCategory();

				bw.write(testingFiles[j] + "\t" + bestCategory+"\n");

			}
			bw.flush();
			bw.close();
		}
	}
}
