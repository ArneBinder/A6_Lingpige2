
import com.aliasi.classify.*;
import com.aliasi.lm.NGramProcessLM;
import com.aliasi.util.AbstractExternalizable;
import com.aliasi.util.Files;

import java.io.*;

public class ClassifyNews {

	private static File[] TRAINING_DIR
			//= new File("../../data/fourNewsGroups/4news-train");
	        //= new File("C:\\Users\\Arne\\Developing\\lib\\lingpipe-4.1.0\\demos\\data\\fourNewsGroups - Kopie\\4news-train");
	        = {new File("C:\\Studium Informatik\\11.FS\\Text Analytics\\A6_SpamFilter\\mailData\\train"),
			new File("C:\\Studium Informatik\\11.FS\\Text Analytics\\A6_SpamFilter\\mailData\\train")};

	private static File TRAIN_HAM = new File("");
	private static File TRAIN_SPAM = new File("");

	private static File TESTING_DIR
			//=  new File("../../data/fourNewsGroups/4news-test");
			//=  new File("C:\\Users\\Arne\\Developing\\lib\\lingpipe-4.1.0\\demos\\data\\fourNewsGroups - Kopie\\4news-test");
			= new File("C:\\Studium Informatik\\11.FS\\Text Analytics\\A6_SpamFilter\\mailData\\test");

	private static String[] CATEGORIES
			/*= { "soc.religion.christian",
			"talk.religion.misc",
			"alt.atheism",
			"misc.forsale" };*/
			= {"mailspam", "mailham"};

	private static int NGRAM_SIZE = 6;

	public static void main(String[] args)
			throws ClassNotFoundException, IOException {

		DynamicLMClassifier<NGramProcessLM> classifier
				= DynamicLMClassifier.createNGramProcess(CATEGORIES, NGRAM_SIZE);

		if (args[0].equals("learn"))
		{
			TRAINING_DIR[1] = new File(args[2]);
			TRAINING_DIR[0] = new File(args[1]);

			for(int i = 0; i < CATEGORIES.length; ++i)
			{
				File classDir = TRAINING_DIR[i];
				if (!classDir.isDirectory()) {
					String msg = "Could not find training directory="
							+ classDir
							+ "\nHave you unpacked 4 newsgroups?";
					System.out.println(msg); // in case exception gets lost in shell
					throw new IllegalArgumentException(msg);
				}

				String[] trainingFiles = classDir.list();
				for (int j = 0; j < trainingFiles.length; ++j) {
					File file = new File(classDir,trainingFiles[j]);
					String text = Files.readFromFile(file,"ISO-8859-1");
					System.out.println("Training on " + CATEGORIES[i] + "/" + trainingFiles[j]);
					Classification classification = new Classification(CATEGORIES[i]);
					Classified<CharSequence> classified = new Classified<CharSequence>(text,classification);
					classifier.handle(classified);
				}

				DataOutputStream outputStream;
				outputStream = new DataOutputStream(new BufferedOutputStream(new FileOutputStream(new File(CATEGORIES[i]+".model"))));
				classifier.languageModel(CATEGORIES[i]).writeTo(outputStream);
			}

		}
		else
		if (args[0].equals("classify"))
		{
			TESTING_DIR = new File(args[1]);

			//classifier.languageModel(CATEGORIES[0]).
			//compiling
			System.out.println("Compiling");
			@SuppressWarnings("unchecked") // we created object so know it's safe
					JointClassifier<CharSequence> compiledClassifier
					= (JointClassifier<CharSequence>) AbstractExternalizable.compile(classifier);

			boolean storeCategories = true;
			JointClassifierEvaluator<CharSequence> evaluator
					= new JointClassifierEvaluator<CharSequence>(compiledClassifier,
					CATEGORIES,
					storeCategories);
			for(int i = 0; i < CATEGORIES.length; ++i) {
				File classDir = new File(TESTING_DIR, CATEGORIES[i]);
				String[] testingFiles = classDir.list();
				for (int j = 0; j < testingFiles.length; ++j) {
					String text = Files.readFromFile(new File(classDir, testingFiles[j]), "ISO-8859-1");
					System.out.print("Testing on " + CATEGORIES[i] + "/" + testingFiles[j] + " ");
					Classification classification = new Classification(CATEGORIES[i]);
					Classified<CharSequence> classified = new Classified<CharSequence>(text, classification);
					evaluator.handle(classified);
					JointClassification jc = compiledClassifier.classify(text);
					String bestCategory = jc.bestCategory();
					String details = jc.toString();
					System.out.println("Got best category of: " + bestCategory);
					System.out.println(jc.toString());
					System.out.println("---------------");
				}
			}

			ConfusionMatrix confMatrix = evaluator.confusionMatrix();
			System.out.println("Total Accuracy: " + confMatrix.totalAccuracy());

			System.out.println("\nFULL EVAL");
			System.out.println(evaluator);
		}



	}
}
